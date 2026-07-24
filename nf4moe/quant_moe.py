"""
4-bit (nf4) quantization for GLM-5.2's fused-3D MoE experts — the part no off-the-shelf
weight-only quantizer (torchao / bnb) reaches, because the experts are stored as fused 3D
`nn.Parameter` (`gate_up_proj` [E, 2*inter, hidden], `down_proj` [E, hidden, inter]) used via
`F.linear(x, w[expert_idx])`, NOT as `nn.Linear` modules.

GLM-5.2 = 743B total, ~700B of which is these experts → bf16 (~1.4TB) does NOT fit on 7xB200
(1.28TB). FP8 fits but has no backward kernel. NVFP4-weight-only is differentiable but skips
the 3D experts. So we quantize the experts ourselves to nf4 and dequantize each routed expert's
2D weight on the fly in forward. Dequant is CONSTANT w.r.t. the activations, so `F.linear(x, w)`
stays differentiable in x → the projector's gradient still flows through the frozen base.

The non-expert params (attention, dense MLPs, shared experts, router, embeddings, lm_head ~ 39B)
stay bf16 (~78GB, fits easily) — only the 700B experts need shrinking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes.functional as bnbF


class QuantizedNaiveMoe(nn.Module):
    """Behavior-preserving drop-in for transformers `GlmMoeDsaNaiveMoe`, experts held as nf4.

    Same forward(hidden_states, top_k_index, top_k_weights) as the original; the ONLY change is
    that `self.gate_up_proj[e]` / `self.down_proj[e]` are replaced by an on-the-fly nf4 dequant of
    expert e's 2D weight. Expert weights are frozen (we never need their grad), so storing them
    quantized loses nothing for training the projector.
    """

    QUANT_TYPE = "nf4"
    BLOCKSIZE = 64

    def __init__(self, num_experts, hidden_dim, intermediate_dim, act_fn,
                 compute_dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.compute_dtype = compute_dtype
        # QuantState objects (hold absmax etc.) — not buffers; moved with the module via _apply.
        self._gate_up_states: list = []
        self._down_states: list = []

    # ---- construction ---------------------------------------------------------------------
    @classmethod
    def from_bf16(cls, gate_up_3d: torch.Tensor, down_3d: torch.Tensor, act_fn, device,
                  compute_dtype=torch.bfloat16) -> "QuantizedNaiveMoe":
        """Quantize the donor module's two 3D expert tensors to per-expert nf4 on `device`.

        gate_up_3d: [E, 2*inter, hidden]   down_3d: [E, hidden, inter]
        """
        num_experts = gate_up_3d.shape[0]
        intermediate_dim = down_3d.shape[2]
        hidden_dim = down_3d.shape[1]
        m = cls(num_experts, hidden_dim, intermediate_dim, act_fn, compute_dtype)
        for e in range(num_experts):
            gu = gate_up_3d[e].to(device=device, dtype=torch.bfloat16).contiguous()
            dn = down_3d[e].to(device=device, dtype=torch.bfloat16).contiguous()
            gup, gus = bnbF.quantize_4bit(gu, blocksize=cls.BLOCKSIZE, quant_type=cls.QUANT_TYPE)
            dnp, dns = bnbF.quantize_4bit(dn, blocksize=cls.BLOCKSIZE, quant_type=cls.QUANT_TYPE)
            m.register_buffer(f"gate_up_packed_{e}", gup, persistent=True)
            m.register_buffer(f"down_packed_{e}", dnp, persistent=True)
            m._gate_up_states.append(gus)
            m._down_states.append(dns)
        return m

    # ---- keep QuantState tensors on the module's device ----------------------------------
    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for st in (*self._gate_up_states, *self._down_states):
            if st is not None:
                st.absmax = fn(st.absmax)
                if getattr(st, "state2", None) is not None and st.state2.absmax is not None:
                    st.state2.absmax = fn(st.state2.absmax)
        return self

    # ---- forward (mirrors GlmMoeDsaNaiveMoe.forward exactly) ------------------------------
    def _deq(self, packed, state):
        # constant w.r.t. activations → F.linear(x, w) below stays differentiable in x
        return bnbF.dequantize_4bit(packed, state, quant_type=self.QUANT_TYPE).to(self.compute_dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            # PERF (audit #18): one .tolist() sync instead of int(expert_idx) + GPU-tensor iteration
            # per hit expert per layer per step (~thousands of GPU->CPU stalls/step). Numerics identical.
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()

        for e in expert_hit:
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            gate_up_w = self._deq(getattr(self, f"gate_up_packed_{e}"), self._gate_up_states[e])
            gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            down_w = self._deq(getattr(self, f"down_packed_{e}"), self._down_states[e])
            current_hidden_states = nn.functional.linear(current_hidden_states, down_w)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx,
                                           current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    @torch.no_grad()
    def memory_bytes(self) -> int:
        tot = 0
        for name, buf in self.named_buffers():
            tot += buf.numel() * buf.element_size()
        for st in (*self._gate_up_states, *self._down_states):
            if st is not None and st.absmax is not None:
                tot += st.absmax.numel() * st.absmax.element_size()
        return tot
