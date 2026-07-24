"""
Load GLM-5.2 (glm_moe_dsa) with its fused-3D MoE experts quantized to nf4 and the model sharded
across N GPUs — the only way the 743B base fits on 7xB200 AND stays differentiable for projector
training (see quant_moe.py for why off-the-shelf quantizers can't reach these experts).

Flow (avoids GPU OOM during load):
  1. from_pretrained → bf16 on CPU (2TB RAM holds the ~1.4TB model; no GPU pressure yet).
  2. Replace every GlmMoeDsaNaiveMoe with nf4 QuantizedNaiveMoe, quantizing onto its target GPU
     (round-robin over the decoder layers). The bf16 3D expert tensor is dropped right after →
     CPU frees incrementally.
  3. Move the remaining bf16 params (attention, dense MLPs, shared experts, router, norms, embed,
     lm_head ~39B) to their layer's GPU.
  4. accelerate.dispatch_model adds the cross-device hooks so activations hop GPU→GPU in forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nf4moe.quant_moe import QuantizedNaiveMoe


def _naive_moe_cls():
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaNaiveMoe
    return GlmMoeDsaNaiveMoe


def quantize_experts_and_shard(model, devices, dtype=torch.bfloat16):
    """In-place: nf4-quantize every fused-3D expert stack and round-robin decoder layers onto
    `devices` (list like ['cuda:0', ... 'cuda:6']). Returns the dispatched model + its device_map."""
    from accelerate import dispatch_model

    NaiveMoe = _naive_moe_cls()
    n = len(devices)
    layers = model.model.layers
    device_map: dict[str, str] = {}

    # non-layer modules live on the first device (embedding's device = where visual tokens splice in)
    device_map["model.embed_tokens"] = devices[0]
    for extra in ("model.rotary_emb", "model.norm"):
        root = model
        ok = True
        for part in extra.split("."):
            if not hasattr(root, part):
                ok = False
                break
            root = getattr(root, part)
        if ok:
            device_map[extra] = devices[0]
    device_map["lm_head"] = devices[0]

    for i, layer in enumerate(layers):
        dev = devices[i % n]
        device_map[f"model.layers.{i}"] = dev
        moe = getattr(layer.mlp, "experts", None)
        if isinstance(moe, NaiveMoe):
            q = QuantizedNaiveMoe.from_bf16(
                moe.gate_up_proj.data, moe.down_proj.data, moe.act_fn, dev, compute_dtype=dtype)
            layer.mlp.experts = q          # drop the bf16 3D params → CPU frees
        # move the rest of this layer's (still-bf16) params to its GPU; quantized experts already there
        layer.to(dev)

    # embed/norm/lm_head to device0
    model.model.embed_tokens.to(devices[0])
    if hasattr(model.model, "norm"):
        model.model.norm.to(devices[0])
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.to(devices[0])
    model.lm_head.to(devices[0])

    model = dispatch_model(model, device_map=device_map)
    model.hf_device_map = device_map
    return model, device_map


# ---------------------------------------------------------------------------------------------
# nf4 checkpoint save/load — so subsequent runs skip the ~100-min bf16 read + re-quantize and load
# the ~470GB nf4 model directly. The plain state_dict drops bnb QuantState (absmax/code), so we
# serialize each QuantizedNaiveMoe's packed bytes + QuantState explicitly, sharded per MoE module.
# ---------------------------------------------------------------------------------------------
def _state_to_cpu_dict(qs):
    import torch
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in qs.as_dict(packed=True).items()}


def save_nf4_checkpoint(model, path):
    """Persist a quantize_experts_and_shard'd model. Layout: config/, non_expert.pt (bf16 of
    everything except expert packed bytes), experts/<module>.pt (one shard per MoE layer:
    packed nf4 + QuantState per expert). Reload with load_nf4_checkpoint — no bf16/re-quant."""
    import os

    import torch

    os.makedirs(os.path.join(path, "experts"), exist_ok=True)
    expert_buf_names = set()
    n_moe = 0
    for name, mod in model.named_modules():
        if isinstance(mod, QuantizedNaiveMoe):
            rec = {"num_experts": mod.num_experts, "hidden_dim": mod.hidden_dim,
                   "intermediate_dim": mod.intermediate_dim, "gate_up": [], "down": []}
            for e in range(mod.num_experts):
                rec["gate_up"].append((getattr(mod, f"gate_up_packed_{e}").cpu(),
                                       _state_to_cpu_dict(mod._gate_up_states[e])))
                rec["down"].append((getattr(mod, f"down_packed_{e}").cpu(),
                                    _state_to_cpu_dict(mod._down_states[e])))
                expert_buf_names.add(f"{name}.gate_up_packed_{e}")
                expert_buf_names.add(f"{name}.down_packed_{e}")
            torch.save(rec, os.path.join(path, "experts", f"{name}.pt"))
            n_moe += 1
    non_expert = {k: v.cpu() for k, v in model.state_dict().items() if k not in expert_buf_names}
    torch.save(non_expert, os.path.join(path, "non_expert.pt"))
    model.config.save_pretrained(path)
    torch.save({"n_moe": n_moe}, os.path.join(path, "meta.pt"))
    return n_moe


def load_nf4_checkpoint(path, devices, dtype=None):
    """Rebuild the nf4 model from save_nf4_checkpoint output WITHOUT bf16/re-quant, sharded over
    `devices`. Mirrors quantize_experts_and_shard's round-robin placement + dispatch."""
    import os

    import torch
    from accelerate import dispatch_model, init_empty_weights
    from bitsandbytes.functional import QuantState
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = dtype or torch.bfloat16
    NaiveMoe = _naive_moe_cls()
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    with init_empty_weights():                       # meta → never allocates the 1.4TB experts
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    n = len(devices)
    layers = model.model.layers
    device_map = {"model.embed_tokens": devices[0], "lm_head": devices[0]}
    for extra in ("model.rotary_emb", "model.norm"):
        if hasattr(model.model, extra.split(".")[-1]):
            device_map[extra] = devices[0]

    # rebuild each QuantizedNaiveMoe from its shard, placed round-robin like the original sharding
    moe_dev = {}
    for i, layer in enumerate(layers):
        dev = devices[i % n]
        device_map[f"model.layers.{i}"] = dev
        if isinstance(getattr(layer.mlp, "experts", None), NaiveMoe):
            moe_dev[f"model.layers.{i}.mlp.experts"] = (layer, dev)

    experts_dir = os.path.join(path, "experts")
    for mod_name, (layer, dev) in moe_dev.items():
        rec = torch.load(os.path.join(experts_dir, f"{mod_name}.pt"), map_location="cpu", weights_only=False)
        q = QuantizedNaiveMoe(rec["num_experts"], rec["hidden_dim"], rec["intermediate_dim"],
                              layer.mlp.experts.act_fn, compute_dtype=dtype)
        for e in range(rec["num_experts"]):
            gp, gs = rec["gate_up"][e]
            dp, ds = rec["down"][e]
            q.register_buffer(f"gate_up_packed_{e}", gp.to(dev), persistent=True)
            q.register_buffer(f"down_packed_{e}", dp.to(dev), persistent=True)
            q._gate_up_states.append(QuantState.from_dict(gs, device=dev))
            q._down_states.append(QuantState.from_dict(ds, device=dev))
        layer.mlp.experts = q

    non_expert = torch.load(os.path.join(path, "non_expert.pt"), map_location="cpu", weights_only=False)
    # assign=True swaps the meta tensors for the loaded CPU bf16 ones (experts already replaced above)
    model.load_state_dict(non_expert, strict=False, assign=True)
    # move the non-expert (now-CPU) params of each layer to its device
    model.model.embed_tokens.to(devices[0])
    if hasattr(model.model, "norm"):
        model.model.norm.to(devices[0])
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.to(devices[0])
    model.lm_head.to(devices[0])
    for i, layer in enumerate(layers):
        # experts already on device; this moves the bf16 attention/MLP/router/norms
        layer.to(devices[i % n])

    model = dispatch_model(model, device_map=device_map)
    model.hf_device_map = device_map
    return model, device_map
