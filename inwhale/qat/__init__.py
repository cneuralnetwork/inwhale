"""
Quantization-Aware Training (QAT) modules for inwhale.
"""

from .ttq import (
    TTQConv2d,
    TTQFunction,
    TTQLinear,
    TTQQuantizer,
    compute_ttq_scales,
    compute_ttq_threshold,
    ttq_quantize,
)

__all__ = [
    "TTQFunction",
    "TTQQuantizer",
    "TTQLinear",
    "TTQConv2d",
    "ttq_quantize",
    "compute_ttq_threshold",
    "compute_ttq_scales",
]
