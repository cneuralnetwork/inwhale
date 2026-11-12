"""
NF4 (NormalFloat4) and FP4 (4-bit Floating Point) Quantization Formats.

Category 7: LLM-Specific Quantization (Milestone 7).

Overview & Motivation:
---------------------
In Large Language Models (LLMs) such as LLaMA, Mistral, Falcon, and GPT models,
parameters occupy immense memory bandwidth and VRAM (e.g. 70B parameter models require
140 GB in FP16 or 280 GB in FP32). Coarse uniform 4-bit quantization (such as INT4)
suffers substantial degradation when applied to weights because neural network weights
are strictly NOT uniformly distributed: they follow zero-mean Gaussian (normal)
distributions:

    W ~ N(0, sigma^2)

Standard uniform quantization places equally spaced bins across the entire dynamic range,
allocating excessive precision to rare tail values and insufficient resolution to the
dense center where over 95% of weights reside.

To solve this, modern LLM quantization employs specialized 4-bit formats:

1. NormalFloat4 (NF4) [Dettmers et al., QLoRA NeurIPS 2023]:
   Information-theoretically optimal quantile-based 4-bit format for Gaussian weights.
   Constructs 16 levels corresponding to equal probability mass quantiles of N(0, 1):
       q_i = Q_X(p_i), where P(X <= q_i) has equal area per bin.
   NF4 features:
   - Canonical 16-level codebook normalized to [-1.0, 1.0].
   - Exact zero representation (at level index 7) ensuring zero padding, embeddings,
     and sparse parameters are represented without numerical drift.
   - Block-wise absolute maximum (absmax) scaling isolating local dynamic ranges.

2. FP4 (4-bit Floating Point) [OCP Microscaling Formats MXFP4 & NVIDIA NVFP4]:
   Hardware-aligned 4-bit floating point representations:
   - FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit (bias = 1).
     Represents unscaled values: +/- {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
     Standardized by OCP (Open Compute Project) and adopted in NVIDIA Blackwell architectures.
   - FP4 BNB: The bitsandbytes FP4 format with non-linear logarithmic distribution.
   - FP4 E3M0: 1 sign bit, 3 exponent bits, 0 mantissa bit (pure logarithmic power-of-two).

3. Bit-Packing (True 4-Bit Storage):
   Packs two 4-bit indices (nibbles) per uint8 byte, delivering 8x compression over FP32
   and 4x compression over FP16/BF16 in physical memory.

4. Drop-in PyTorch Neural Network Layers:
   - `NF4Linear`: Drop-in replacement for `nn.Linear` with weights stored in packed NF4.
   - `FP4Linear`: Drop-in replacement for `nn.Linear` with weights stored in packed FP4.

References:
-----------
- Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
  arXiv:2305.14314.
- Open Compute Project (OCP). "Microscaling Formats (MX) Specification v1.0." (2023).
- Rouhani et al. (2023). "With Shared Microscaling (MX) Formats, Less is More." arXiv:2310.10537.
"""

from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from inwhale.core.quantizer import BaseQuantizer


# ============================================================================
# Canonical Quantization Codebooks
# ============================================================================

# 16-level Canonical NormalFloat4 (NF4) Codebook (Dettmers et al. QLoRA 2023)
# Derived from N(0, 1) quantiles with exact zero at index 7.
NF4_CANONICAL_CODEBOOK: List[float] = [
    -1.00000000,
    -0.69619280,
    -0.52507305,
    -0.39491749,
    -0.28444138,
    -0.18477343,
    -0.09105004,
    0.00000000,
    0.07958030,
    0.16093020,
    0.24611230,
    0.33791524,
    0.44070983,
    0.56261700,
    0.72295684,
    1.00000000,
]

# Standard OCP MXFP4 / NVFP4 E2M1 Codebook (1 sign, 2 exponent, 1 mantissa, bias=1)
# Unscaled positive levels: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# Normalized to [-1.0, 1.0] by dividing by 6.0.
# Contains 16 levels (with duplicate zero representing +/-0 in standard IEEE float).
FP4_E2M1_CODEBOOK: List[float] = [
    -6.0 / 6.0,   # -1.0
    -4.0 / 6.0,   # -0.6666667
    -3.0 / 6.0,   # -0.5
    -2.0 / 6.0,   # -0.3333333
    -1.5 / 6.0,   # -0.25
    -1.0 / 6.0,   # -0.1666667
    -0.5 / 6.0,   # -0.0833333
    0.0 / 6.0,    # -0.0
    0.0 / 6.0,    # +0.0
    0.5 / 6.0,    # +0.0833333
    1.0 / 6.0,    # +0.1666667
    1.5 / 6.0,    # +0.25
    2.0 / 6.0,    # +0.3333333
    3.0 / 6.0,    # +0.5
    4.0 / 6.0,    # +0.6666667
    6.0 / 6.0,    # +1.0
]

# BitsAndBytes FP4 format codebook (normalized by 12.0)
FP4_BNB_CODEBOOK: List[float] = [
    -12.0 / 12.0,
    -8.0 / 12.0,
    -6.0 / 12.0,
    -4.0 / 12.0,
    -3.0 / 12.0,
    -2.0 / 12.0,
    -0.0625 / 12.0,
    0.0,
    0.0,
    0.0625 / 12.0,
    2.0 / 12.0,
    3.0 / 12.0,
    4.0 / 12.0,
    6.0 / 12.0,
    8.0 / 12.0,
    12.0 / 12.0,
]

# FP4 E3M0 (Pure Logarithmic Power-of-Two: 1 sign, 3 exponent, 0 mantissa, bias=3)
FP4_E3M0_CODEBOOK: List[float] = [
    -16.0 / 16.0,  # 2^4 = 16
    -8.0 / 16.0,   # 2^3 = 8
    -4.0 / 16.0,   # 2^2 = 4
    -2.0 / 16.0,   # 2^1 = 2
    -1.0 / 16.0,   # 2^0 = 1
    -0.5 / 16.0,   # 2^-1 = 0.5
    -0.25 / 16.0,  # 2^-2 = 0.25
    0.0,
    0.0,
    0.25 / 16.0,
    0.5 / 16.0,
    1.0 / 16.0,
    2.0 / 16.0,
    4.0 / 16.0,
    8.0 / 16.0,
    16.0 / 16.0,
]


class FP4Type(str, Enum):
    """Supported 4-bit floating point variations."""
    E2M1 = "e2m1"  # OCP MXFP4 & NVIDIA NVFP4 standard
    BNB = "bnb"    # BitsAndBytes FP4 format
    E3M0 = "e3m0"  # Logarithmic pure power-of-two FP4


# ============================================================================
# Algorithmic Quantile Map Generation
# ============================================================================

def create_normal_map(
    num_bits: int = 4,
    offset: float = 0.9677083,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Algorithmic derivation of the NormalFloat (NF4) codebook from N(0, 1) quantiles.

    Following Dettmers et al. (QLoRA 2023):
    Constructs quantiles with equal probability mass under standard normal distribution:
    - 8 positive quantiles derived from linspace(offset, 0.5, 9)[:-1]
    - 7 negative quantiles derived from -linspace(offset, 0.5, 8)[:-1]
    - 1 exact zero point separating negative and positive intervals (total 16 values).
    - Normalized by max absolute value so endpoints are [-1.0, 1.0].

    Args:
        num_bits: Bit-width of the format (default: 4 for NF4).
        offset: Outermost quantile boundary covering tail standard deviations (default: 0.9677083).
        device: PyTorch device for returned tensor.
        dtype: PyTorch float dtype for returned tensor.

    Returns:
        torch.Tensor of shape (2^num_bits,) containing normalized quantiles.
    """
    if num_bits != 4:
        raise ValueError(f"NormalFloat map currently supports 4-bit (NF4), got num_bits={num_bits}")

    try:
        from scipy.stats import norm
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1].numpy()).tolist()
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1].numpy())).tolist()
        codebook_list = v3 + [0.0] + v1[::-1]
        codebook = torch.tensor(codebook_list, device=device, dtype=dtype)
        codebook = codebook / codebook.abs().max()
        return codebook
    except ImportError:
        # Fallback to high-precision precomputed canonical NF4 codebook
        return torch.tensor(NF4_CANONICAL_CODEBOOK, device=device, dtype=dtype)


def get_codebook(
    format_name: str = "nf4",
    fp4_type: Union[FP4Type, str] = FP4Type.E2M1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Retrieve codebook tensor for specified 4-bit format.

    Args:
        format_name: "nf4" or "fp4".
        fp4_type: When format_name="fp4", specifies variant ("e2m1", "bnb", "e3m0").
        device: Target torch device.
        dtype: Target torch floating-point dtype.

    Returns:
        Monotonically sorted 1D tensor of 16 codebook values in [-1.0, 1.0].
    """
    fmt = format_name.lower().strip()
    if fmt == "nf4":
        return torch.tensor(NF4_CANONICAL_CODEBOOK, device=device, dtype=dtype)
    elif fmt == "fp4":
        if isinstance(fp4_type, str):
            fp4_type = FP4Type(fp4_type.lower().strip())
        if fp4_type == FP4Type.E2M1:
            return torch.tensor(FP4_E2M1_CODEBOOK, device=device, dtype=dtype)
        elif fp4_type == FP4Type.BNB:
            return torch.tensor(FP4_BNB_CODEBOOK, device=device, dtype=dtype)
        elif fp4_type == FP4Type.E3M0:
            return torch.tensor(FP4_E3M0_CODEBOOK, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown FP4 type: {fp4_type}")
    else:
        raise ValueError(f"Unknown format_name: '{format_name}'. Expected 'nf4' or 'fp4'.")


# ============================================================================
# 4-Bit Bit-Packing / Unpacking Utilities
# ============================================================================

def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 4-bit integer indices (values 0..15) into uint8 bytes.

    Two 4-bit nibbles are packed into each byte:
    - low nibble (bits 0..3): element 2*k
    - high nibble (bits 4..7): element 2*k + 1

    This achieves 50% memory compression over uint8 (8x vs FP32, 4x vs FP16).

    Args:
        indices: Integer tensor with values in [0, 15].

    Returns:
        Packed 1D torch.uint8 tensor with ceil(indices.numel() / 2) elements.
    """
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=indices.device)

    flat = indices.reshape(-1).to(torch.uint8)
    numel = flat.numel()

    # Check bounds
    if (flat > 15).any():
        raise ValueError("Indices for 4-bit packing must be in range [0, 15].")

    if numel % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])

    low_nibble = flat[0::2] & 0x0F
    high_nibble = (flat[1::2] & 0x0F) << 4
    packed = low_nibble | high_nibble
    return packed


def unpack_4bit(
    packed: torch.Tensor,
    original_numel: int,
    original_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Unpack uint8 bytes into 4-bit integer indices (0..15).

    Args:
        packed: Packed torch.uint8 tensor.
        original_numel: Number of original unpacked elements.
        original_shape: Optional shape to reshape the unpacked tensor to.

    Returns:
        torch.uint8 tensor containing original unpacked indices.
    """
    if original_numel == 0:
        if original_shape is not None:
            return torch.empty(original_shape, dtype=torch.uint8, device=packed.device)
        return torch.empty(0, dtype=torch.uint8, device=packed.device)

    flat_packed = packed.reshape(-1)
    low_nibble = flat_packed & 0x0F
    high_nibble = (flat_packed >> 4) & 0x0F

    unpacked = torch.empty(flat_packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    unpacked[0::2] = low_nibble
    unpacked[1::2] = high_nibble
    unpacked = unpacked[:original_numel]

    if original_shape is not None:
        unpacked = unpacked.reshape(original_shape)

    return unpacked


# ============================================================================
# Core Functional Quantization and Dequantization
# ============================================================================

def _compute_midpoints(codebook: torch.Tensor) -> torch.Tensor:
    """Compute midpoint boundaries between sorted codebook levels for bucketization."""
    return (codebook[:-1] + codebook[1:]) / 2.0


def quantize_to_codebook(
    x: torch.Tensor,
    codebook: torch.Tensor,
    block_size: int = 64,
    pack: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, ...]]:
    """
    Quantize an arbitrary tensor to a 16-level codebook using block-wise absmax scaling.

    Args:
        x: Input floating point tensor.
        codebook: Monotonically sorted 1D tensor of 16 codebook levels in [-1.0, 1.0].
        block_size: Number of contiguous elements per quantization block (default: 64).
        pack: If True, pack indices into torch.uint8 (2 indices per byte).
        eps: Small epsilon to prevent division by zero in zero-valued blocks.

    Returns:
        Tuple of:
        - quantized_indices: torch.uint8 tensor of indices (or packed uint8 if pack=True)
        - scales: block scales of shape (num_blocks, 1)
        - pad_len: number of zero-padded elements appended
        - original_shape: shape tuple of input tensor
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    original_shape = tuple(x.shape)
    numel = x.numel()
    if numel == 0:
        empty_indices = torch.empty(0, dtype=torch.uint8, device=x.device)
        empty_scales = torch.empty(0, dtype=x.dtype, device=x.device)
        return empty_indices, empty_scales, 0, original_shape

    flat_x = x.reshape(-1)

    # Pad to multiple of block_size if needed
    remainder = numel % block_size
    pad_len = (block_size - remainder) % block_size
    if pad_len > 0:
        flat_x = F.pad(flat_x, (0, pad_len), value=0.0)

    blocks = flat_x.reshape(-1, block_size)

    # Compute block-wise absmax scale: s_b = max(|W_b|)
    # Since codebook max is 1.0, normalized x is W_b / s_b in [-1.0, 1.0]
    scales = torch.max(blocks.abs(), dim=-1, keepdim=True).values
    scales = torch.clamp(scales, min=eps)

    # Normalize blocks to [-1.0, 1.0]
    normalized_blocks = blocks / scales
    normalized_blocks = torch.clamp(normalized_blocks, -1.0, 1.0)

    # Fast bucketization via precomputed midpoints
    midpoints = _compute_midpoints(codebook.to(device=x.device, dtype=normalized_blocks.dtype))
    indices = torch.bucketize(normalized_blocks, midpoints).to(torch.uint8)

    if pack:
        packed_indices = pack_4bit(indices)
        return packed_indices, scales, pad_len, original_shape
    else:
        # Unpad indices back to original numel and reshape to original_shape
        indices_flat = indices.reshape(-1)[:numel]
        indices_orig = indices_flat.reshape(original_shape)
        return indices_orig, scales, pad_len, original_shape


def dequantize_from_codebook(
    indices: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    block_size: int = 64,
    pad_len: int = 0,
    original_shape: Optional[Tuple[int, ...]] = None,
    is_packed: bool = False,
) -> torch.Tensor:
    """
    Dequantize 4-bit indices back to floating point using codebook and block scales.

    Args:
        indices: Quantized indices (unpacked uint8 or packed uint8).
        scales: Block-wise scales of shape (num_blocks, 1).
        codebook: Codebook tensor with 16 entries in [-1.0, 1.0].
        block_size: Number of elements per block.
        pad_len: Number of padding elements to strip.
        original_shape: Shape to reconstruct output to.
        is_packed: Set to True if indices are packed 4-bit in uint8.

    Returns:
        Reconstructed floating point tensor matching original_shape and scale dtype.
    """
    if indices.numel() == 0:
        if original_shape is not None:
            return torch.empty(original_shape, dtype=scales.dtype, device=scales.device)
        return torch.empty(0, dtype=scales.dtype, device=scales.device)

    device = scales.device
    dtype = scales.dtype
    cb = codebook.to(device=device, dtype=dtype)

    if is_packed:
        total_elements = scales.shape[0] * block_size
        flat_indices = unpack_4bit(indices, total_elements).long()
        blocks_indices = flat_indices.reshape(scales.shape[0], block_size)
    else:
        flat_indices = indices.reshape(-1).long()
        if pad_len > 0:
            flat_indices = F.pad(flat_indices, (0, pad_len), value=7)
        blocks_indices = flat_indices.reshape(scales.shape[0], block_size)

    # Dequantize: X_hat = codebook[indices] * scale
    dequantized_blocks = cb[blocks_indices] * scales
    flat_dequantized = dequantized_blocks.reshape(-1)

    if pad_len > 0:
        flat_dequantized = flat_dequantized[:-pad_len]

    if original_shape is not None:
        return flat_dequantized.reshape(original_shape)
    return flat_dequantized


# ============================================================================
# Autograd Straight-Through Estimator (STE) for QAT
# ============================================================================

class _CodebookFakeQuantizeSTE(torch.autograd.Function):
    """Straight-Through Estimator (STE) function for codebook fake quantization."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, codebook: torch.Tensor, block_size: int, eps: float):
        indices, scales, pad_len, orig_shape = quantize_to_codebook(
            x, codebook, block_size=block_size, pack=False, eps=eps
        )
        x_hat = dequantize_from_codebook(
            indices, scales, codebook, block_size=block_size, pad_len=pad_len, original_shape=orig_shape
        )
        return x_hat

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: pass gradients directly through quantization function
        return grad_output, None, None, None


# ============================================================================
# High-Level Functional APIs for NF4 and FP4
# ============================================================================

def nf4_quantize(
    x: torch.Tensor,
    block_size: int = 64,
    pack: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, ...]]:
    """
    Quantize tensor to 4-bit NormalFloat (NF4).

    Args:
        x: Input floating point tensor.
        block_size: Block size for absmax scaling (default: 64, standard QLoRA).
        pack: If True, pack 4-bit indices into uint8.
        eps: Small epsilon for numerical stability.

    Returns:
        (indices, scales, pad_len, original_shape)
    """
    codebook = get_codebook("nf4", device=x.device, dtype=x.dtype)
    return quantize_to_codebook(x, codebook, block_size=block_size, pack=pack, eps=eps)


def nf4_dequantize(
    indices: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 64,
    pad_len: int = 0,
    shape: Optional[Tuple[int, ...]] = None,
    is_packed: bool = False,
) -> torch.Tensor:
    """
    Dequantize 4-bit NormalFloat (NF4) indices back to floating point.

    Args:
        indices: Quantized NF4 indices (unpacked or packed uint8).
        scales: Block-wise scales.
        block_size: Block size (default: 64).
        pad_len: Padding length to strip.
        shape: Output shape.
        is_packed: True if indices are packed 4-bit.

    Returns:
        Dequantized floating point tensor.
    """
    codebook = get_codebook("nf4", device=scales.device, dtype=scales.dtype)
    return dequantize_from_codebook(
        indices, scales, codebook, block_size=block_size, pad_len=pad_len, original_shape=shape, is_packed=is_packed
    )


def nf4_fake_quantize(
    x: torch.Tensor,
    block_size: int = 64,
    eps: float = 1e-8,
    use_ste: bool = True,
) -> torch.Tensor:
    """
    Simulate end-to-end 4-bit NormalFloat (NF4) quantization and dequantization.

    Args:
        x: Input tensor.
        block_size: Block size for absmax scaling (default: 64).
        eps: Small epsilon.
        use_ste: If True, uses Straight-Through Estimator for autograd backpropagation.

    Returns:
        Quantized and dequantized tensor with identical shape and dtype to x.
    """
    codebook = get_codebook("nf4", device=x.device, dtype=x.dtype)
    if use_ste and x.requires_grad:
        return _CodebookFakeQuantizeSTE.apply(x, codebook, block_size, eps)
    else:
        indices, scales, pad_len, orig_shape = quantize_to_codebook(
            x, codebook, block_size=block_size, pack=False, eps=eps
        )
        return dequantize_from_codebook(
            indices, scales, codebook, block_size=block_size, pad_len=pad_len, original_shape=orig_shape
        )


def fp4_quantize(
    x: torch.Tensor,
    block_size: int = 32,
    fp4_type: Union[FP4Type, str] = FP4Type.E2M1,
    pack: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, ...]]:
    """
    Quantize tensor to 4-bit Floating Point (FP4).

    Args:
        x: Input floating point tensor.
        block_size: Block size for microscaling (default: 32, standard OCP MXFP4).
        fp4_type: Format variant ("e2m1", "bnb", "e3m0").
        pack: If True, pack 4-bit indices into uint8.
        eps: Small epsilon for numerical stability.

    Returns:
        (indices, scales, pad_len, original_shape)
    """
    codebook = get_codebook("fp4", fp4_type=fp4_type, device=x.device, dtype=x.dtype)
    return quantize_to_codebook(x, codebook, block_size=block_size, pack=pack, eps=eps)


def fp4_dequantize(
    indices: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 32,
    fp4_type: Union[FP4Type, str] = FP4Type.E2M1,
    pad_len: int = 0,
    shape: Optional[Tuple[int, ...]] = None,
    is_packed: bool = False,
) -> torch.Tensor:
    """
    Dequantize 4-bit Floating Point (FP4) indices back to floating point.

    Args:
        indices: Quantized FP4 indices.
        scales: Block-wise scales.
        block_size: Block size (default: 32).
        fp4_type: Format variant.
        pad_len: Padding length.
        shape: Output shape.
        is_packed: True if indices are packed.

    Returns:
        Dequantized floating point tensor.
    """
    codebook = get_codebook("fp4", fp4_type=fp4_type, device=scales.device, dtype=scales.dtype)
    return dequantize_from_codebook(
        indices, scales, codebook, block_size=block_size, pad_len=pad_len, original_shape=shape, is_packed=is_packed
    )


def fp4_fake_quantize(
    x: torch.Tensor,
    block_size: int = 32,
    fp4_type: Union[FP4Type, str] = FP4Type.E2M1,
    eps: float = 1e-8,
    use_ste: bool = True,
) -> torch.Tensor:
    """
    Simulate end-to-end 4-bit Floating Point (FP4) quantization and dequantization.

    Args:
        x: Input tensor.
        block_size: Block size for microscaling (default: 32).
        fp4_type: Format variant ("e2m1", "bnb", "e3m0").
        eps: Small epsilon.
        use_ste: If True, uses Straight-Through Estimator for autograd.

    Returns:
        Quantized and dequantized tensor.
    """
    codebook = get_codebook("fp4", fp4_type=fp4_type, device=x.device, dtype=x.dtype)
    if use_ste and x.requires_grad:
        return _CodebookFakeQuantizeSTE.apply(x, codebook, block_size, eps)
    else:
        indices, scales, pad_len, orig_shape = quantize_to_codebook(
            x, codebook, block_size=block_size, pack=False, eps=eps
        )
        return dequantize_from_codebook(
            indices, scales, codebook, block_size=block_size, pad_len=pad_len, original_shape=orig_shape
        )


# ============================================================================
# Object-Oriented Quantizers (Inheriting BaseQuantizer)
# ============================================================================

class NF4Quantizer(BaseQuantizer):
    """
    Object-Oriented 4-bit NormalFloat (NF4) Quantizer.

    Quantizes normally-distributed weights into 16 optimal quantiles with block-wise
    absmax scaling and optional true 4-bit memory packing.
    """

    def __init__(
        self,
        block_size: int = 64,
        pack: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__(bits=4)
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")

        self.block_size = block_size
        self.pack = pack
        self.eps = eps

        # State tracked across quantize / dequantize cycles
        self.scales: Optional[torch.Tensor] = None
        self.pad_len: int = 0
        self.original_shape: Optional[Tuple[int, ...]] = None
        self.is_packed: bool = pack

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor x using NF4 format."""
        indices, scales, pad_len, orig_shape = nf4_quantize(
            x, block_size=self.block_size, pack=self.pack, eps=self.eps
        )
        self.scales = scales
        self.pad_len = pad_len
        self.original_shape = orig_shape
        self.is_packed = self.pack
        return indices

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to floating point using stored scales and shape."""
        if self.scales is None:
            raise RuntimeError("Cannot dequantize before quantize() has been called or scales set.")

        return nf4_dequantize(
            indices,
            self.scales,
            block_size=self.block_size,
            pad_len=self.pad_len,
            shape=self.original_shape,
            is_packed=self.is_packed,
        )

    def fake_quantize(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        """Simulate NF4 quantization and dequantization end-to-end."""
        return nf4_fake_quantize(x, block_size=self.block_size, eps=self.eps, use_ste=use_ste)


class FP4Quantizer(BaseQuantizer):
    """
    Object-Oriented 4-bit Floating Point (FP4) Quantizer.

    Supports OCP MXFP4 (E2M1), bitsandbytes FP4 (BNB), and logarithmic power-of-two (E3M0).
    """

    def __init__(
        self,
        block_size: int = 32,
        fp4_type: Union[FP4Type, str] = FP4Type.E2M1,
        pack: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__(bits=4)
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")

        if isinstance(fp4_type, str):
            fp4_type = FP4Type(fp4_type.lower().strip())

        self.block_size = block_size
        self.fp4_type = fp4_type
        self.pack = pack
        self.eps = eps

        self.scales: Optional[torch.Tensor] = None
        self.pad_len: int = 0
        self.original_shape: Optional[Tuple[int, ...]] = None
        self.is_packed: bool = pack

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor x using FP4 format."""
        indices, scales, pad_len, orig_shape = fp4_quantize(
            x, block_size=self.block_size, fp4_type=self.fp4_type, pack=self.pack, eps=self.eps
        )
        self.scales = scales
        self.pad_len = pad_len
        self.original_shape = orig_shape
        self.is_packed = self.pack
        return indices

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to floating point using stored scales and shape."""
        if self.scales is None:
            raise RuntimeError("Cannot dequantize before quantize() has been called or scales set.")

        return fp4_dequantize(
            indices,
            self.scales,
            block_size=self.block_size,
            fp4_type=self.fp4_type,
            pad_len=self.pad_len,
            shape=self.original_shape,
            is_packed=self.is_packed,
        )

    def fake_quantize(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        """Simulate FP4 quantization and dequantization end-to-end."""
        return fp4_fake_quantize(
            x, block_size=self.block_size, fp4_type=self.fp4_type, eps=self.eps, use_ste=use_ste
        )


# ============================================================================
# Drop-in PyTorch Neural Network Modules
# ============================================================================

class NF4Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear storing weights in 4-bit NormalFloat (NF4).

    Features:
    - True 4-bit memory packing (packed uint8 buffers, 2 weights per byte).
    - Block-wise absmax scaling (default block size: 64).
    - Dynamic on-the-fly dequantization during forward pass.
    - `from_linear` factory for instant conversion of pretrained model layers.
    - `memory_footprint` inspection utility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 64,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.weight_shape = (out_features, in_features)

        # Initialize full precision weight and quantize to packed NF4 buffer
        init_weight = torch.randn(out_features, in_features, device=device, dtype=dtype) * (1.0 / math.sqrt(in_features))
        packed_w, scales, pad_len, _ = nf4_quantize(init_weight, block_size=block_size, pack=True)

        self.register_buffer("weight_packed", packed_w)
        self.register_buffer("weight_scales", scales)
        self.pad_len = pad_len

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = 64) -> "NF4Linear":
        """
        Convert a standard torch.nn.Linear layer to an NF4Linear layer.

        Args:
            linear: Source nn.Linear module.
            block_size: Block size for absmax quantization (default: 64).

        Returns:
            NF4Linear module with identical weights quantized into NF4.
        """
        device = linear.weight.device
        dtype = linear.weight.dtype

        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )

        # Quantize original weights
        packed_w, scales, pad_len, _ = nf4_quantize(linear.weight.data, block_size=block_size, pack=True)
        mod.weight_packed.copy_(packed_w)
        mod.weight_scales.copy_(scales)
        mod.pad_len = pad_len

        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.data)

        return mod

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize packed NF4 weights into standard floating point tensor."""
        return nf4_dequantize(
            self.weight_packed,
            self.weight_scales,
            block_size=self.block_size,
            pad_len=self.pad_len,
            shape=self.weight_shape,
            is_packed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic on-the-fly NF4 dequantization."""
        w = self.dequantize_weight().to(dtype=x.dtype, device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, w, bias)

    def memory_footprint(self) -> Dict[str, Union[int, float]]:
        """
        Calculate memory consumption and compression ratio of layer parameters.

        Returns:
            Dictionary with byte counts and compression ratio vs FP32 and FP16.
        """
        numel = self.in_features * self.out_features
        packed_bytes = self.weight_packed.numel() * self.weight_packed.element_size()
        scales_bytes = self.weight_scales.numel() * self.weight_scales.element_size()
        bias_bytes = (self.bias.numel() * self.bias.element_size()) if self.bias is not None else 0
        total_quantized_bytes = packed_bytes + scales_bytes + bias_bytes

        fp32_bytes = numel * 4 + bias_bytes
        fp16_bytes = numel * 2 + bias_bytes

        return {
            "num_parameters": numel,
            "quantized_bytes": total_quantized_bytes,
            "fp32_baseline_bytes": fp32_bytes,
            "fp16_baseline_bytes": fp16_bytes,
            "compression_vs_fp32": fp32_bytes / max(total_quantized_bytes, 1),
            "compression_vs_fp16": fp16_bytes / max(total_quantized_bytes, 1),
        }


class FP4Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear storing weights in 4-bit Floating Point (FP4).

    Features:
    - OCP MXFP4 / NVFP4 (E2M1) microscaling.
    - True 4-bit memory packing.
    - Dynamic on-the-fly dequantization during forward pass.
    - `from_linear` factory.
    - `memory_footprint` inspection utility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 32,
        fp4_type: Union[FP4Type, str] = FP4Type.E2M1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.fp4_type = FP4Type(fp4_type.lower().strip()) if isinstance(fp4_type, str) else fp4_type
        self.weight_shape = (out_features, in_features)

        init_weight = torch.randn(out_features, in_features, device=device, dtype=dtype) * (1.0 / math.sqrt(in_features))
        packed_w, scales, pad_len, _ = fp4_quantize(
            init_weight, block_size=block_size, fp4_type=self.fp4_type, pack=True
        )

        self.register_buffer("weight_packed", packed_w)
        self.register_buffer("weight_scales", scales)
        self.pad_len = pad_len

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 32,
        fp4_type: Union[FP4Type, str] = FP4Type.E2M1,
    ) -> "FP4Linear":
        """
        Convert a standard torch.nn.Linear layer to an FP4Linear layer.

        Args:
            linear: Source nn.Linear module.
            block_size: Block size for microscaling (default: 32).
            fp4_type: Format variant ("e2m1", "bnb", "e3m0").

        Returns:
            FP4Linear module.
        """
        device = linear.weight.device
        dtype = linear.weight.dtype

        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            fp4_type=fp4_type,
            device=device,
            dtype=dtype,
        )

        packed_w, scales, pad_len, _ = fp4_quantize(
            linear.weight.data, block_size=block_size, fp4_type=mod.fp4_type, pack=True
        )
        mod.weight_packed.copy_(packed_w)
        mod.weight_scales.copy_(scales)
        mod.pad_len = pad_len

        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.data)

        return mod

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize packed FP4 weights into standard floating point tensor."""
        return fp4_dequantize(
            self.weight_packed,
            self.weight_scales,
            block_size=self.block_size,
            fp4_type=self.fp4_type,
            pad_len=self.pad_len,
            shape=self.weight_shape,
            is_packed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic on-the-fly FP4 dequantization."""
        w = self.dequantize_weight().to(dtype=x.dtype, device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, w, bias)

    def memory_footprint(self) -> Dict[str, Union[int, float]]:
        """Calculate parameter memory consumption and compression metrics."""
        numel = self.in_features * self.out_features
        packed_bytes = self.weight_packed.numel() * self.weight_packed.element_size()
        scales_bytes = self.weight_scales.numel() * self.weight_scales.element_size()
        bias_bytes = (self.bias.numel() * self.bias.element_size()) if self.bias is not None else 0
        total_quantized_bytes = packed_bytes + scales_bytes + bias_bytes

        fp32_bytes = numel * 4 + bias_bytes
        fp16_bytes = numel * 2 + bias_bytes

        return {
            "num_parameters": numel,
            "quantized_bytes": total_quantized_bytes,
            "fp32_baseline_bytes": fp32_bytes,
            "fp16_baseline_bytes": fp16_bytes,
            "compression_vs_fp32": fp32_bytes / max(total_quantized_bytes, 1),
            "compression_vs_fp16": fp16_bytes / max(total_quantized_bytes, 1),
        }
