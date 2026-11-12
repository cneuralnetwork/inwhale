"""
LLM-specific quantization algorithms and utilities.

Category 7: LLM-Specific Quantization (Milestone 7).
"""

from inwhale.llm.nf4_fp4 import (
    NF4_CANONICAL_CODEBOOK,
    FP4_E2M1_CODEBOOK,
    FP4_BNB_CODEBOOK,
    FP4_E3M0_CODEBOOK,
    FP4Type,
    create_normal_map,
    get_codebook,
    pack_4bit,
    unpack_4bit,
    quantize_to_codebook,
    dequantize_from_codebook,
    nf4_quantize,
    nf4_dequantize,
    nf4_fake_quantize,
    fp4_quantize,
    fp4_dequantize,
    fp4_fake_quantize,
    NF4Quantizer,
    FP4Quantizer,
    NF4Linear,
    FP4Linear,
)

__all__ = [
    "NF4_CANONICAL_CODEBOOK",
    "FP4_E2M1_CODEBOOK",
    "FP4_BNB_CODEBOOK",
    "FP4_E3M0_CODEBOOK",
    "FP4Type",
    "create_normal_map",
    "get_codebook",
    "pack_4bit",
    "unpack_4bit",
    "quantize_to_codebook",
    "dequantize_from_codebook",
    "nf4_quantize",
    "nf4_dequantize",
    "nf4_fake_quantize",
    "fp4_quantize",
    "fp4_dequantize",
    "fp4_fake_quantize",
    "NF4Quantizer",
    "FP4Quantizer",
    "NF4Linear",
    "FP4Linear",
]
