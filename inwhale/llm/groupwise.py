"""
Group-wise (Block-wise) Quantization for Large Language Models (LLMs).

Category 7: LLM-Specific Quantization (Milestone 7).

Overview & Motivation:
---------------------
In Large Language Models (e.g. LLaMA, Mistral, Falcon, GPT architectures),
weight matrices have vast channel dimensions (e.g., hidden size 4096, intermediate
size 14336). Features and parameters frequently exhibit extreme numerical outliers
concentrated in specific channels or feature coordinates (Dettmers et al., 2022;
Lin et al., 2023; Frantar et al., 2022).

Standard coarse quantization schemes suffer from outlier pollution:
1. Per-Tensor Quantization: A single outlier scales the entire matrix, destroying
   numerical resolution for all other normal values.
2. Per-Channel Quantization: While isolating channels, a single outlier still degrades
   all elements (e.g., 4096+ weights) along that channel.

Group-wise (Block-wise) Quantization resolves this by partitioning each channel or
tensor along a target axis into small, contiguous blocks/groups (typically of size
G in {32, 64, 128, 256}). Each group computes its own independent scale factor and
zero-point:

    W: [--- Group 0 ---|--- Group 1 ---|--- Group 2 ---| ... ]
          scale_0           scale_1           scale_2
          zp_0              zp_1              zp_2

Consequently, an outlier is strictly isolated to its small G-element group. The
remaining groups retain ultra-fine dynamic ranges and minimal quantization error.
This is the foundational quantization scheme employed in production LLM inference
and post-training algorithms, including GPTQ, AWQ, and QLoRA.

Mathematical Formulation:
-------------------------
For an input tensor X partitioned into groups of size G along dimension `dim`:

1. Symmetric Group-wise Quantization:
   Given bit-width b:
       qmax = 2^(b - 1) - 1
       qmin = -2^(b - 1)
   For each group g:
       scale_g = max(max_{i in g} |X_i| / qmax, eps)
       Q_i = clamp(round(X_i / scale_g), qmin, qmax)
   Dequantization:
       X_hat_i = Q_i * scale_g

2. Asymmetric Group-wise Quantization:
   Given bit-width b:
       qmin = 0, qmax = 2^b - 1
   For each group g:
       min_g = min_{i in g} X_i
       max_g = max_{i in g} X_i
       scale_g = max((max_g - min_g) / (qmax - qmin), eps)
       zp_g = round(qmin - min_g / scale_g)
       Q_i = clamp(round(X_i / scale_g + zp_g), qmin, qmax)
   Dequantization:
       X_hat_i = (Q_i - zp_g) * scale_g

References:
-----------
- Frantar et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers." arXiv:2210.17323.
- Lin et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression
  and Acceleration." MLSys 2024.
- Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from inwhale.core.quantizer import BaseQuantizer
from inwhale.rounding.base import RoundingStrategy
from inwhale.rounding.nearest import NearestRounding


def _pad_along_dim(x: torch.Tensor, dim: int, group_size: int) -> Tuple[torch.Tensor, int]:
    """
    Pad tensor along `dim` so that its size is an exact multiple of `group_size`.
    Pads by repeating the last boundary element so that min/max ranges of the partial
    group are preserved without artificial distortion.

    Args:
        x: Input tensor.
        dim: Dimension along which to pad.
        group_size: Target group size.

    Returns:
        Tuple of (padded_tensor, pad_len).
    """
    dim = dim if dim >= 0 else dim + x.dim()
    length = x.shape[dim]
    if length <= group_size:
        return x, 0

    remainder = length % group_size
    if remainder == 0:
        return x, 0

    pad_len = group_size - remainder

    # Slice the last element along dim and replicate it pad_len times
    last_slice = x.select(dim, -1).unsqueeze(dim)
    repeat_shape = [1] * x.dim()
    repeat_shape[dim] = pad_len
    pad_tensor = last_slice.repeat(*repeat_shape)

    padded = torch.cat([x, pad_tensor], dim=dim)
    return padded, pad_len


def groupwise_quantize(
    x: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    dim: int = -1,
    symmetric: bool = True,
    rounding: Optional[RoundingStrategy] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """
    Quantize an arbitrary tensor using group-wise (block-wise) quantization.

    Args:
        x: Input tensor of arbitrary shape.
        bits: Number of quantization bits (e.g. 2, 4, 8).
        group_size: Number of elements in each quantization block.
        dim: Axis along which groups are formed. Defaults to -1.
        symmetric: If True, uses symmetric signed quantization around 0.
            If False, uses asymmetric unsigned quantization with zero-point.
        rounding: Rounding strategy instance. Defaults to NearestRounding().
        eps: Small epsilon preventing division by zero.

    Returns:
        Tuple of:
            - qx: Quantized discrete integer tensor of same shape as x.
            - scale: Group-wise scale tensor.
            - zero_point: Group-wise zero-point tensor (or None if symmetric).
            - pad_len: Number of padded elements added along dim.
    """
    if bits < 1:
        raise ValueError(f"Bits must be >= 1, got {bits}.")
    if group_size < 1:
        raise ValueError(f"Group size must be >= 1, got {group_size}.")
    if eps <= 0:
        raise ValueError(f"Epsilon must be positive, got {eps}.")

    if x.numel() == 0:
        empty_scale = torch.empty(0, dtype=x.dtype, device=x.device)
        return x.clone(), empty_scale, None if symmetric else empty_scale, 0

    if rounding is None:
        rounding = NearestRounding()

    # Normalize dim index
    norm_dim = dim if dim >= 0 else dim + x.dim()
    if not (0 <= norm_dim < x.dim()):
        raise IndexError(f"Dimension out of range: dim={dim} for tensor of shape {x.shape}")

    # Determine integer quantization grid
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))
    else:
        qmin = 0
        qmax = (1 << bits) - 1

    orig_length = x.shape[norm_dim]
    effective_group_size = min(group_size, orig_length)

    # Pad along target dimension if not divisible by effective_group_size
    x_padded, pad_len = _pad_along_dim(x, norm_dim, effective_group_size)

    # Permute target dimension to the last axis for group reshaping
    x_perm = x_padded.movedim(norm_dim, -1)
    num_groups = x_perm.shape[-1] // effective_group_size

    # Reshape to (*batch_shape, num_groups, effective_group_size)
    x_grouped = x_perm.unflatten(-1, (num_groups, effective_group_size))

    if symmetric:
        max_abs = x_grouped.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(max_abs / qmax, min=eps)
        zero_point = None

        qx_grouped = x_grouped / scale
        qx_grouped = rounding.round(qx_grouped)
        qx_grouped = torch.clamp(qx_grouped, qmin, qmax)
    else:
        min_val = x_grouped.amin(dim=-1, keepdim=True)
        max_val = x_grouped.amax(dim=-1, keepdim=True)
        range_val = max_val - min_val

        # Constant group handling
        zero_mask = range_val < eps
        safe_scale = torch.where(
            zero_mask,
            torch.clamp(max_val.abs() / qmax, min=eps),
            range_val / (qmax - qmin),
        )
        scale = torch.clamp(safe_scale, min=eps)

        zp_real = qmin - min_val / scale
        zero_point = torch.where(
            zero_mask,
            torch.full_like(scale, qmin),
            rounding.round(zp_real),
        )

        qx_grouped = x_grouped / scale + zero_point
        qx_grouped = rounding.round(qx_grouped)
        qx_grouped = torch.clamp(qx_grouped, qmin, qmax)

    # Flatten back from groups and remove padding
    qx_perm = qx_grouped.flatten(-2, -1)
    if pad_len > 0:
        qx_perm = qx_perm[..., :orig_length]

    # Restore original dimension order
    qx = qx_perm.movedim(-1, norm_dim)
    return qx, scale, zero_point, pad_len


def groupwise_dequantize(
    qx: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    group_size: int = 128,
    dim: int = -1,
    pad_len: int = 0,
) -> torch.Tensor:
    """
    Dequantize a group-wise quantized tensor back to continuous floating point.

    Args:
        qx: Quantized discrete tensor.
        scale: Scale tensor from groupwise_quantize.
        zero_point: Zero-point tensor (or None if symmetric).
        group_size: Group size used during quantization.
        dim: Dimension along which groups were formed.
        pad_len: Padding length returned by groupwise_quantize.

    Returns:
        Dequantized floating point tensor with identical shape and device as qx.
    """
    if qx.numel() == 0:
        return qx.clone().float()

    norm_dim = dim if dim >= 0 else dim + qx.dim()
    orig_length = qx.shape[norm_dim]
    effective_group_size = min(group_size, orig_length)

    # If padding was applied, reconstruct the padded shape for group arithmetic
    if pad_len > 0:
        qx_padded, _ = _pad_along_dim(qx, norm_dim, effective_group_size)
    else:
        qx_padded = qx

    qx_perm = qx_padded.movedim(norm_dim, -1)
    num_groups = qx_perm.shape[-1] // effective_group_size
    qx_grouped = qx_perm.unflatten(-1, (num_groups, effective_group_size))

    scale = scale.to(device=qx.device, dtype=qx.dtype if qx.is_floating_point() else torch.float32)

    if zero_point is None:
        rec_grouped = qx_grouped * scale
    else:
        zp = zero_point.to(device=qx.device, dtype=scale.dtype)
        rec_grouped = (qx_grouped - zp) * scale

    rec_perm = rec_grouped.flatten(-2, -1)
    if pad_len > 0:
        rec_perm = rec_perm[..., :orig_length]

    rec = rec_perm.movedim(-1, norm_dim)
    return rec


def groupwise_fake_quantize(
    x: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    dim: int = -1,
    symmetric: bool = True,
    rounding: Optional[RoundingStrategy] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Simulate group-wise quantization and dequantization in a single step (fake quantization).

    Args:
        x: Input floating-point tensor.
        bits: Number of quantization bits.
        group_size: Group/block size.
        dim: Axis along which groups are formed.
        symmetric: Symmetric vs asymmetric mode.
        rounding: Rounding strategy instance.
        eps: Numerical stability constant.

    Returns:
        Reconstructed tensor x_hat approximating x under group-wise quantization.
    """
    qx, scale, zero_point, pad_len = groupwise_quantize(
        x=x,
        bits=bits,
        group_size=group_size,
        dim=dim,
        symmetric=symmetric,
        rounding=rounding,
        eps=eps,
    )
    return groupwise_dequantize(
        qx=qx,
        scale=scale,
        zero_point=zero_point,
        group_size=group_size,
        dim=dim,
        pad_len=pad_len,
    )


class GroupwiseQuantizer(BaseQuantizer):
    """
    Group-wise Quantizer for LLMs (subclasses BaseQuantizer).

    Provides an object-oriented stateful quantizer that conforms to the BaseQuantizer
    interface while maintaining internal scale and zero-point parameters.

    Attributes:
        bits: Number of quantization bits (e.g. 4 for INT4, 8 for INT8).
        group_size: Number of elements in each quantization block.
        dim: Axis along which groups are formed.
        symmetric: Symmetric (signed) or Asymmetric (unsigned with zero-point) mode.
        rounding: Rounding strategy.
        eps: Small constant preventing division by zero.
        scale: Computed group-wise scale tensor.
        zero_point: Computed group-wise zero-point tensor.
        pad_len: Number of padded elements added.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        dim: int = -1,
        symmetric: bool = True,
        rounding: Optional[RoundingStrategy] = None,
        eps: float = 1e-8,
    ):
        if bits < 1:
            raise ValueError(f"Bits must be >= 1, got {bits}.")
        if group_size < 1:
            raise ValueError(f"Group size must be >= 1, got {group_size}.")
        if eps <= 0:
            raise ValueError(f"Epsilon must be positive, got {eps}.")

        super().__init__(bits=bits)

        self.group_size = group_size
        self.dim = dim
        self.symmetric = symmetric
        self.rounding = rounding if rounding is not None else NearestRounding()
        self.eps = eps

        if symmetric:
            self.qmax = (1 << (bits - 1)) - 1
            self.qmin = -(1 << (bits - 1))
        else:
            self.qmin = 0
            self.qmax = (1 << bits) - 1

        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None
        self.pad_len: int = 0

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor into discrete integer levels and cache scale/zero-point.

        Args:
            x: Input continuous tensor.

        Returns:
            Quantized integer tensor with identical shape to x.
        """
        qx, scale, zero_point, pad_len = groupwise_quantize(
            x=x,
            bits=self.bits,
            group_size=self.group_size,
            dim=self.dim,
            symmetric=self.symmetric,
            rounding=self.rounding,
            eps=self.eps,
        )
        self.scale = scale
        self.zero_point = zero_point
        self.pad_len = pad_len
        return qx

    def dequantize(self, qx: torch.Tensor) -> torch.Tensor:
        """
        Dequantize integer tensor using cached scale and zero-point parameters.

        Args:
            qx: Quantized integer tensor.

        Returns:
            Reconstructed floating-point tensor.
        """
        if self.scale is None:
            raise RuntimeError("Quantizer has not quantized any tensor yet. Call quantize() first.")

        return groupwise_dequantize(
            qx=qx,
            scale=self.scale,
            zero_point=self.zero_point,
            group_size=self.group_size,
            dim=self.dim,
            pad_len=self.pad_len,
        )

    def fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run quantize and dequantize consecutively.
        """
        qx = self.quantize(x)
        return self.dequantize(qx)


class GroupwiseLinear(nn.Module):
    """
    Drop-in Linear layer with Group-wise Quantized weights.

    Stores quantized weights and group-wise scale factors (and optional zero-points),
    dequantizing on-the-fly during the forward pass to perform matrix multiplication:
        y = x @ W_rec^T + bias

    This mirrors production LLM architectures (e.g. AutoGPTQ, ExLlama, AWQ) where
    weights are compressed to 4 bits or 8 bits in groups of 64 or 128 elements.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias.
        bits: Bit-width for weight quantization (default: 4).
        group_size: Number of elements per group along in_features (default: 128).
        symmetric: If True, uses symmetric INT4/INT8 quantization.
        device: Target torch device.
        dtype: Target torch data type.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric

        # Weight is quantized along in_features (dim = -1 or 1 in shape (out_features, in_features))
        self.quantizer = GroupwiseQuantizer(
            bits=bits,
            group_size=group_size,
            dim=1,
            symmetric=symmetric,
        )

        # Buffers for quantized weights and scaling parameters
        self.register_buffer("qweight", torch.zeros(out_features, in_features, device=device, dtype=torch.float32))
        num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer("scales", torch.ones(out_features, num_groups, 1, device=device, dtype=dtype or torch.float32))
        if not symmetric:
            self.register_buffer("zero_points", torch.zeros(out_features, num_groups, 1, device=device, dtype=dtype or torch.float32))
        else:
            self.zero_points = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype or torch.float32))
        else:
            self.register_parameter("bias", None)

        self.pad_len = (group_size - (in_features % group_size)) % group_size if in_features % group_size != 0 else 0

    def set_weights(self, weight: torch.Tensor):
        """
        Quantize and store floating-point weight tensor W in shape (out_features, in_features).
        """
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Expected weight shape ({self.out_features}, {self.in_features}), got {weight.shape}."
            )
        qx = self.quantizer.quantize(weight)
        self.qweight.copy_(qx)
        self.scales.copy_(self.quantizer.scale)
        if not self.symmetric and self.quantizer.zero_point is not None:
            self.zero_points.copy_(self.quantizer.zero_point)
        self.pad_len = self.quantizer.pad_len

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
    ) -> "GroupwiseLinear":
        """
        Construct a GroupwiseLinear module by quantizing an existing nn.Linear layer.

        Args:
            linear: PyTorch nn.Linear layer to quantize.
            bits: Bit-width (default: 4).
            group_size: Quantization group size (default: 128).
            symmetric: Whether to use symmetric quantization.

        Returns:
            Instantiated and quantized GroupwiseLinear layer.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        layer.set_weights(linear.weight.data)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def get_dequantized_weight(self) -> torch.Tensor:
        """
        Reconstruct full-precision weight tensor from quantized buffer and scale parameters.
        """
        return groupwise_dequantize(
            qx=self.qweight,
            scale=self.scales,
            zero_point=self.zero_points,
            group_size=self.group_size,
            dim=1,
            pad_len=self.pad_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: dequantizes weight and executes linear projection.
        """
        weight_deq = self.get_dequantized_weight()
        return F.linear(x, weight_deq, self.bias)

    def memory_savings(self) -> dict:
        """
        Calculate memory consumption and compression ratio vs FP32.

        Returns:
            Dictionary with parameter counts, memory in bytes, and compression ratio.
        """
        num_elements = self.in_features * self.out_features
        num_groups = (self.in_features + self.group_size - 1) // self.group_size
        total_groups = self.out_features * num_groups

        fp32_bytes = num_elements * 4.0
        # Bits per quantized weight + 32-bit float scale per group
        qweight_bytes = (num_elements * self.bits) / 8.0
        scale_bytes = total_groups * 4.0
        zp_bytes = total_groups * 4.0 if not self.symmetric else 0.0
        bias_bytes = self.out_features * 4.0 if self.bias is not None else 0.0

        quantized_bytes = qweight_bytes + scale_bytes + zp_bytes + bias_bytes
        compression_ratio = fp32_bytes / max(quantized_bytes, 1.0)

        return {
            "num_elements": num_elements,
            "bits": self.bits,
            "group_size": self.group_size,
            "fp32_bytes": fp32_bytes,
            "quantized_bytes": quantized_bytes,
            "compression_ratio": compression_ratio,
        }
