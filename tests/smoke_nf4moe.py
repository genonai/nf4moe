"""Real-model validation of the nf4-MoE path on the actual 743B GLM-5.2 (isolated: no vision, no
Trainer). Proves the thing tiny synthetic tests can't: the REAL fused-3D experts quantize, the
model fits on the GPUs (no CPU spill), and a trainable input-side module's gradient flows through
the frozen nf4 base. Needs: the GLM-5.2 bf16 checkpoint, ~2 TB host RAM, 4+ big GPUs.
"""
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM

from nf4moe import quantize_experts_and_shard

PATH = "zai-org/GLM-5.2"


class Projector(nn.Sequential):
    """Stand-in for any trainable input-side module (e.g. a VLM projector): in_dim → hidden MLP."""

    def __init__(self, in_dim, hidden):
        super().__init__(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))


def main():
    ngpu = torch.cuda.device_count()
    print(f"[1] loading bf16 -> CPU (~1.4TB) ... visible GPUs={ngpu}", flush=True)
    llm = AutoModelForCausalLM.from_pretrained(
        PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)

    print("[2] nf4-quantizing fused-3D experts + sharding across GPUs ...", flush=True)
    llm, dmap = quantize_experts_and_shard(llm, [f"cuda:{i}" for i in range(ngpu)], dtype=torch.bfloat16)

    bad = {str(d) for d in dmap.values() if str(d) in ("cpu", "disk", "meta")}
    assert not bad, f"device_map spilled to {bad} — base still did not fit"
    print(f"[2] sharded OK, no CPU/meta spill. layers on devices: "
          f"{sorted(set(str(dmap[k]) for k in dmap if 'layers.' in k))}", flush=True)
    for i in range(ngpu):
        print(f"    cuda:{i} weights = {torch.cuda.memory_allocated(i)/1024**3:.1f} GiB", flush=True)

    for p in llm.parameters():
        p.requires_grad_(False)
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dev0 = llm.get_input_embeddings().weight.device
    proj = Projector(4096, llm.config.hidden_size).to(dev0, torch.bfloat16)
    for p in proj.parameters():
        p.requires_grad_(True)

    # dummy "visual tokens" (donor-side dim 4096) spliced into placeholder positions
    n = 8
    visual = proj(torch.randn(n, 4096, device=dev0, dtype=torch.bfloat16))
    ids = torch.tensor([[1, 2, 3] + [0] * n + [4, 5]], device=dev0)
    emb = llm.get_input_embeddings()(ids).clone()
    emb[ids == 0] = visual.to(emb.dtype)
    labels = torch.tensor([[-100] * (3 + n) + [4, 5]], device=dev0)

    print("[3] forward ...", flush=True)
    out = llm(inputs_embeds=emb, attention_mask=torch.ones_like(ids), labels=labels)
    print(f"[3] loss={out.loss.item():.4f} finite={torch.isfinite(out.loss).item()}", flush=True)

    print("[4] backward ...", flush=True)
    out.loss.backward()
    grads = [p.grad for p in proj.parameters() if p.grad is not None]
    gn = (sum((x.float() ** 2).sum() for x in grads)).sqrt().item() if grads else 0.0
    finite = all(torch.isfinite(x).all().item() for x in grads)
    print(f"[4] projector grad tensors={len(grads)} grad_norm={gn:.4e} finite={finite}", flush=True)
    print("NF4MOE BACKWARD:", "PASS ✅" if grads and gn > 0 and finite else "BLOCKED ❌", flush=True)


if __name__ == "__main__":
    main()
