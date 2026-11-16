"""
LLM-specific quantization algorithms and utilities.

Category 7: LLM-Specific Quantization (Milestone 7).
"""

from inwhale.llm.mixed_precision import (
    MixedPrecisionState,
    find_outlier_channels,
    detect_outliers_by_quantile,
    split_activation_channels,
    split_weight_channels,
    quantize_int8_vector_wise,
    dequantize_int8_vector_wise,
    quantize_int8_tensor_wise,
    mixed_precision_matmul,
    mixed_precision_matmul_ste,
    MixedPrecisionActivationQuantizer,
    MixedPrecisionLinear,
    compare_mixed_vs_uniform_8bit,
)

__all__ = [
    "MixedPrecisionState",
    "find_outlier_channels",
    "detect_outliers_by_quantile",
    "split_activation_channels",
    "split_weight_channels",
    "quantize_int8_vector_wise",
    "dequantize_int8_vector_wise",
    "quantize_int8_tensor_wise",
    "mixed_precision_matmul",
    "mixed_precision_matmul_ste",
    "MixedPrecisionActivationQuantizer",
    "MixedPrecisionLinear",
    "compare_mixed_vs_uniform_8bit",
]
