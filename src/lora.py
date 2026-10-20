"""
LoRA: Low-Rank Adaptation of Large Language Models
Paper:     https://arxiv.org/abs/2106.09685 (Hu et al., Microsoft 2021)
Notes:     /Users/chaitanya/Development/AI/research-papers/lora/lora-notes.md
Reference: Microsoft original — github.com/microsoft/LoRA/blob/main/loralib/layers.py
           HuggingFace PEFT  — github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py

Core math:
    During fine-tuning, instead of updating all weights W, freeze W₀ and learn:

        ΔW = B × A      where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r ≪ min(d, k)

    Forward pass:
        h = W₀x + (α/r) · B(A(dropout(x)))

    Initialization (matches both Microsoft and HuggingFace reference implementations):
        A ~ kaiming_uniform (a=sqrt(5))  — same as PyTorch default nn.Linear init
        B = zeros                         — ensures ΔW = 0 at fine-tuning start

Usage:
    from src.lora import apply_lora, lora_param_count, get_lora_state_dict, merge_lora

    model = MiniGPT(...)

    # --- Switch ON ---
    apply_lora(model, rank=4, alpha=16, enabled=True)

    # --- Switch OFF (full fine-tuning) ---
    apply_lora(model, enabled=False)

    # After fine-tuning: fuse adapters into base weights → zero inference overhead
    merge_lora(model)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core LoRA layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with trainable low-rank adapter matrices A and B.

    Forward pass:
        h = W₀x + (α/r) · B(A(dropout(x)))

    Attributes:
        linear   : the original frozen weight W₀
        lora_A   : compresses input  (in_features → rank)
        lora_B   : expands to output (rank → out_features)
        scale    : α/r — keeps magnitude consistent when changing rank
        dropout  : optional regularisation on the LoRA input path
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.linear  = linear           # frozen — not updated during training
        self.rank    = rank
        self.scale   = alpha / rank     # α/r scaling factor

        in_features  = linear.in_features
        out_features = linear.out_features
        device       = linear.weight.device   # match device of the frozen weight (e.g. cuda:0)

        # Adapter matrices — the only things that train
        self.lora_A = nn.Linear(in_features,  rank,         bias=False).to(device)
        self.lora_B = nn.Linear(rank,         out_features, bias=False).to(device)

        # Initialization — verified against Microsoft and HuggingFace reference implementations:
        #   A: kaiming_uniform (a=sqrt(5)) = PyTorch's own default nn.Linear init
        #   B: zeros → ΔW = B×A = 0 at start, model begins from pretrained behaviour
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Optional dropout applied to x before lora_A (Microsoft's placement)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen path: original pretrained computation
        base_out = self.linear(x)
        # LoRA path: low-rank correction
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return base_out + self.scale * lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA into base weight: W = W₀ + (α/r) · B×A
        Returns a plain nn.Linear — zero inference overhead after merging.
        """
        merged = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=False,
            device=self.linear.weight.device,
        )
        delta = self.scale * (self.lora_B.weight @ self.lora_A.weight)
        merged.weight = nn.Parameter(self.linear.weight.data + delta)
        return merged


# ---------------------------------------------------------------------------
# Switch: apply / remove LoRA
# ---------------------------------------------------------------------------

def apply_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 16.0,
    target_modules: tuple[str, ...] = ("W_v", "W_o"),
    dropout_p: float = 0.0,
    enabled: bool = True,
) -> nn.Module:
    """
    Inject LoRA adapters and freeze everything else.

    Args:
        model:          MiniGPT model to adapt.
        rank:           Rank r of decomposition. Paper recommends 1–8.
                        r=4 is the paper's default — small r often matches full fine-tuning.
        alpha:          Scaling factor α. Effective scale = α/r.
                        Setting alpha=rank gives scale=1; alpha=2×rank gives scale=2.
        target_modules: Attribute names of nn.Linear layers to wrap.
                        Default ("W_v", "W_o") — chosen for compatibility with QK Norm.
                        The LoRA paper recommends ("W_q", "W_v") for standard GPT, but our
                        enhanced model applies rms_norm to Q and K after projection (QK Norm).
                        This rescales W_q's output to unit RMS, washing out the LoRA correction
                        on W_q. W_v and W_o have no such normalization, so their adapters flow
                        through unchanged — giving LoRA its full effective capacity.
        dropout_p:      Dropout probability on the LoRA input path (0.0 = disabled).
        enabled:        Master switch. False = skip LoRA, all params remain trainable.

    Returns:
        Model modified in-place (and returned for chaining).
    """
    if not enabled:
        print("LoRA disabled — all parameters will train (full fine-tuning mode).")
        return model

    replaced = 0
    for module in model.modules():
        for attr_name in target_modules:
            if hasattr(module, attr_name):
                original = getattr(module, attr_name)
                if isinstance(original, nn.Linear):
                    setattr(module, attr_name, LoRALinear(original, rank, alpha, dropout_p))
                    replaced += 1

    # Freeze everything except LoRA adapter matrices
    for name, param in model.named_parameters():
        param.requires_grad = ("lora_A" in name or "lora_B" in name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"LoRA enabled — {replaced} layers replaced with rank-{rank} adapters "
          f"(α={alpha}, scale={alpha/rank:.2f}).")
    print(f"  Trainable : {trainable:>10,}  ({100 * trainable / total:.2f}%)")
    print(f"  Frozen    : {total - trainable:>10,}")
    print(f"  Total     : {total:>10,}")
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """
    Fuse all LoRA adapters into base weights: W = W₀ + (α/r)·BA
    Replaces every LoRALinear with a plain nn.Linear.
    Unfreezes all parameters after merging (ready for further full fine-tuning if needed).
    """
    merged_count = 0
    for module in model.modules():
        # _modules is how nn.Module stores registered submodules (setattr goes here)
        for attr_name, child in list(module._modules.items()):
            if isinstance(child, LoRALinear):
                setattr(module, attr_name, child.merge_weights())
                merged_count += 1

    for param in model.parameters():
        param.requires_grad = True

    print(f"LoRA merged — {merged_count} adapters fused into base weights. "
          f"Zero inference overhead.")
    return model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def lora_param_count(model: nn.Module) -> dict:
    """Print and return a breakdown of trainable vs frozen parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total     = trainable + frozen
    pct       = round(100 * trainable / total, 3) if total > 0 else 0.0
    print(f"  Trainable : {trainable:>12,}  ({pct}%)")
    print(f"  Frozen    : {frozen:>12,}")
    print(f"  Total     : {total:>12,}")
    return {"trainable": trainable, "frozen": frozen, "total": total, "pct": pct}


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Return only the LoRA adapter weights (lora_A and lora_B matrices).
    Use this to save a tiny fine-tuned checkpoint — MBs instead of hundreds of MBs.
    Load alongside the base model checkpoint to reconstruct the fine-tuned model.
    """
    lora_sd = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    size_mb = sum(v.numel() * v.element_size() for v in lora_sd.values()) / 1e6
    print(f"LoRA state dict: {len(lora_sd)} tensors, ~{size_mb:.2f} MB")
    return lora_sd
