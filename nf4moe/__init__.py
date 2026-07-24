"""nf4moe — QLoRA-compatible NF4 quantization for fused-3D MoE expert tensors.

Modern `transformers` MoE checkpoints store experts as fused 3D ``nn.Parameter``s
(``gate_up_proj [E, 2*inter, hidden]``, ``down_proj [E, hidden, inter]``), not
``nn.Linear`` modules — so every module-swap weight-only quantizer (bitsandbytes,
torchao, AWQ, GPTQ) silently skips them. This package quantizes those experts to
NF4 by hand and dequantizes only the *routed* experts on the fly in forward.
Dequant is constant w.r.t. the activations, so the frozen base stays
differentiable and gradients flow through to any trainable input-side module
(projector, LoRA) — i.e. standard QLoRA, ported to 3D expert tensors.

Validated on GLM-5.2 (743B total / 39B active): ~1.49 TB bf16 → ~414 GB,
finite loss and projector gradients through the full frozen base.
"""

from nf4moe.quant_moe import QuantizedNaiveMoe
from nf4moe.load_quant import (
    quantize_experts_and_shard,
    save_nf4_checkpoint,
    load_nf4_checkpoint,
)

__all__ = [
    "QuantizedNaiveMoe",
    "quantize_experts_and_shard",
    "save_nf4_checkpoint",
    "load_nf4_checkpoint",
]
