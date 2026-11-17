"""
LLM-specific quantization algorithms and utilities.

Category 7: LLM-Specific Quantization.
"""

from inwhale.llm.outlier_aware import (
    SmoothQuantState,
    OutlierDiagnostics,
    compute_smoothquant_scales,
    apply_activation_smoothing,
    apply_weight_smoothing,
    fold_smoothing_into_layernorm,
    quantize_activation_per_token,
    dequantize_activation_per_token,
    quantize_weight_per_channel,
    dequantize_weight_per_channel,
    clip_activation_outliers,
    analyze_activation_outliers,
    SmoothQuantObserver,
    smoothquant_gemm,
    smoothquant_linear_ste,
    SmoothQuantQuantizer,
    SmoothQuantLinear,
    compare_smoothquant_vs_standard_8bit,
)

__all__ = [
    "SmoothQuantState",
    "OutlierDiagnostics",
    "compute_smoothquant_scales",
    "apply_activation_smoothing",
    "apply_weight_smoothing",
    "fold_smoothing_into_layernorm",
    "quantize_activation_per_token",
    "dequantize_activation_per_token",
    "quantize_weight_per_channel",
    "dequantize_weight_per_channel",
    "clip_activation_outliers",
    "analyze_activation_outliers",
    "SmoothQuantObserver",
    "smoothquant_gemm",
    "smoothquant_linear_ste",
    "SmoothQuantQuantizer",
    "SmoothQuantLinear",
    "compare_smoothquant_vs_standard_8bit",
]
