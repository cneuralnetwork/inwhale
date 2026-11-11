"""
Per-Row and Per-Column Quantization for Large Language Models (LLMs).

Category 7: LLM-Specific Quantization (Milestone 7).

Overview & Motivation:
---------------------
In Large Language Models (e.g., LLaMA, Mistral, GPT, Falcon architectures), linear
projections in multi-head attention (Q, K, V, O) and feed-forward networks (gate, up, down)
dominate model parameters and memory footprints.

Standard coarse quantization schemes suffer from outlier degradation:
1. Per-Tensor Quantization:
   A single scale factor is computed for the entire matrix. When systematic outliers
   emerge (features with magnitudes up to 100x higher than normal coordinates), the
   quantization bin width expands, crushing normal values into zero or near-zero bins.
2. Per-Row Quantization (Row-wise):
   Each row of a 2D matrix (or along the row dimension of multidimensional tensors)
   maintains an independent scale and zero-point.
   - For weight matrices W in PyTorch shape (out_features, in_features), each row
     corresponds to an output channel / output neuron.
   - For activation tensors X in shape (num_tokens, hidden_dim), each row corresponds
     to a distinct token vector. Token activations vary wildly across sequence positions
     (e.g., delimiter tokens vs content tokens). Per-row quantization preserves
     token-level dynamic range (Per-Token Quantization).
3. Per-Column Quantization (Column-wise):
   Each column of a 2D matrix maintains an independent scale and zero-point.
   - For weight matrices W, each column corresponds to an input feature / channel.
     Empirical findings (Dettmers et al., 2022 LLM.int8(); Xiao et al., 2023 SmoothQuant)
     demonstrate that activation outliers are strictly column-aligned. Column-wise
     quantization isolates input-channel-specific variances.
4. Vector-Wise INT8 / INT4 GEMM (Dettmers et al., 2022):
   Combining per-row activation quantization with per-column weight quantization allows
   pure integer matrix multiplication followed by an outer-product scale reconstruction:
       Y = (Q_A @ Q_W) * (s_A @ s_W)
   with zero accuracy loss and maximal throughput.

Mathematical Formulation:
-------------------------
For an input tensor X in shape (..., M, N):

1. Per-Row Quantization:
   Preserves row dimension M, reducing over column dimension N (dim=-1, keepdim=True).
   - Symmetric (bit-width b, qmin = -2^(b-1), qmax = 2^(b-1) - 1):
       scale_i = max(max_j |X_{i, j}| / qmax, eps)
       Q_{i, j} = clamp(round(X_{i, j} / scale_i), qmin, qmax)
       X_hat_{i, j} = Q_{i, j} * scale_i
   - Asymmetric (bit-width b, qmin = 0, qmax = 2^b - 1):
       scale_i = max((max_j X_{i, j} - min_j X_{i, j}) / (qmax - qmin), eps)
       zp_i = clamp(round(qmin - min_j X_{i, j} / scale_i), qmin, qmax)
       Q_{i, j} = clamp(round(X_{i, j} / scale_i + zp_i), qmin, qmax)
       X_hat_{i, j} = (Q_{i, j} - zp_i) * scale_i

2. Per-Column Quantization:
   Preserves column dimension N, reducing over row dimension M (dim=-2, keepdim=True).
   - Symmetric:
       scale_j = max(max_i |X_{i, j}| / qmax, eps)
       Q_{i, j} = clamp(round(X_{i, j} / scale_j), qmin, qmax)
       X_hat_{i, j} = Q_{i, j} * scale_j
   - Asymmetric:
       scale_j = max((max_i X_{i, j} - min_i X_{i, j}) / (qmax - qmin), eps)
       zp_j = clamp(round(qmin - min_i X_{i, j} / scale_j), qmin, qmax)
       Q_{i, j} = clamp(round(X_{i, j} / scale_j + zp_j), qmin, qmax)
       X_hat_{i, j} = (Q_{i, j} - zp_j) * scale_j

References:
-----------
- Dettmers et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale."
  NeurIPS 2022. arXiv:2208.07339.
- Xiao et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models."
  ICML 2023. arXiv:2211.10438.
- Yao et al. (2022). "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers."
  NeurIPS 2022. arXiv:2206.01861.
- Lin et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration."
  MLSys 2024. arXiv:2306.00978.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from inwhale.core.quantizer import BaseQuantizer
from inwhale.rounding.base import RoundingStrategy
from inwhale.rounding.nearest import NearestRounding


def _get_qmin_qmax(
    bits: int, symmetric: bool, signed: Optional[bool] = None
) -> Tuple[int, int]:
    """
    Compute integer quantization bounds [qmin, qmax] for a given bit-width.
    """
    if bits < 1:
        raise ValueError(f"Bits must be >= 1, got {bits}.")

    if symmetric:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
    else:
        is_signed = False if signed is None else signed
        if is_signed:
            qmin = -(1 << (bits - 1))
            qmax = (1 << (bits - 1)) - 1
        else:
            qmin = 0
            qmax = (1 << bits) - 1

    return qmin, qmax


def per_row_quantize(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    signed: Optional[bool] = None,
    rounding: Optional[RoundingStrategy] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize input tensor using per-row quantization.

    For a 2D matrix (M, N), each row i in [0, M-1] receives an independent scale factor
    (and zero-point if asymmetric). For multidimensional tensors (..., M, N), the
    reduction is performed over the last dimension (dim=-1), keeping row dimension M.

    Args:
        x: Input tensor (e.g., shape (M, N) or (..., M, N)).
        bits: Number of quantization bits (e.g., 4, 8). Defaults to 8.
        symmetric: If True, uses symmetric quantization around 0.
            If False, uses asymmetric quantization with zero-point. Defaults to True.
        signed: Whether asymmetric quantization uses signed integers.
            Ignored if symmetric is True. Defaults to False.
        rounding: Rounding strategy instance. Defaults to NearestRounding().
        eps: Minimum threshold to prevent division by zero in scale computation.

    Returns:
        Tuple of (qx, scale, zero_point):
            - qx: Quantized integer tensor with identical shape and device as x.
            - scale: Row scale tensor of shape (..., M, 1).
            - zero_point: Row zero-point tensor of shape (..., M, 1), or None if symmetric.
    """
    if x.numel() == 0:
        return x.clone(), torch.empty(0, device=x.device, dtype=x.dtype), None

    if rounding is None:
        rounding = NearestRounding()

    # Dimension to reduce: for 1D, reduce dim 0; for >= 2D, reduce dim -1 (columns)
    reduce_dim = -1 if x.dim() >= 2 else 0

    if bits == 1 and symmetric:
        # Binary 1-bit signed quantization: values map to {-1, +1}
        max_abs = torch.amax(torch.abs(x), dim=reduce_dim, keepdim=True)
        scale = torch.clamp(max_abs, min=eps)
        qx = torch.sign(x)
        qx[qx == 0] = 1.0
        return qx, scale, None

    qmin, qmax = _get_qmin_qmax(bits=bits, symmetric=symmetric, signed=signed)

    if symmetric:
        max_abs = torch.amax(torch.abs(x), dim=reduce_dim, keepdim=True)
        scale = torch.clamp(max_abs / qmax, min=eps)
        qx = x / scale
        qx = rounding.round(qx)
        qx = torch.clamp(qx, qmin, qmax)
        return qx, scale, None
    else:
        min_val = torch.amin(x, dim=reduce_dim, keepdim=True)
        max_val = torch.amax(x, dim=reduce_dim, keepdim=True)
        # Ensure 0.0 is represented in the dynamic range so that zero-point
        # maps cleanly without saturation clipping
        zero_t = torch.zeros(1, dtype=x.dtype, device=x.device)
        min_val = torch.minimum(min_val, zero_t)
        max_val = torch.maximum(max_val, zero_t)

        scale = torch.clamp((max_val - min_val) / float(qmax - qmin), min=eps)
        zp_real = float(qmin) - min_val / scale
        zero_point = torch.clamp(rounding.round(zp_real), qmin, qmax)

        qx = x / scale + zero_point
        qx = rounding.round(qx)
        qx = torch.clamp(qx, qmin, qmax)
        return qx, scale, zero_point


def per_row_dequantize(
    qx: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dequantize an integer tensor back to continuous floating-point values using per-row scales.

    Args:
        qx: Quantized tensor.
        scale: Per-row scale tensor (shape broadcastable to qx).
        zero_point: Optional per-row zero-point tensor (for asymmetric quantization).

    Returns:
        Dequantized floating-point tensor.
    """
    if qx.numel() == 0:
        return qx.clone()

    if zero_point is None:
        return qx * scale
    return (qx - zero_point) * scale


def per_row_fake_quantize(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    signed: Optional[bool] = None,
    rounding: Optional[RoundingStrategy] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Execute per-row quantization followed immediately by dequantization (fake quantization).

    Args:
        x: Input tensor.
        bits: Number of quantization bits.
        symmetric: Symmetric or asymmetric mode.
        signed: Signed integer mode for asymmetric quantization.
        rounding: Rounding strategy instance.
        eps: Minimum scale epsilon.

    Returns:
        Reconstructed floating-point tensor with simulated per-row quantization error.
    """
    qx, scale, zero_point = per_row_quantize(
        x=x,
        bits=bits,
        symmetric=symmetric,
        signed=signed,
        rounding=rounding,
        eps=eps,
    )
    return per_row_dequantize(qx=qx, scale=scale, zero_point=zero_point)


def per_column_quantize(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    signed: Optional[bool] = None,
    rounding: Optional[RoundingStrategy] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize input tensor using per-column quantization.

    For a 2D matrix (M, N), each column j in [0, N-1] receives an independent scale factor
    (and zero-point if asymmetric). For multidimensional tensors (..., M, N), the
    reduction is performed over the second-to-last dimension (dim=-2), keeping column dimension N.

    Args:
        x: Input tensor (e.g., shape (M, N) or (..., M, N)).
        bits: Number of quantization bits (e.g., 4, 8). Defaults to 8.
        symmetric: If True, uses symmetric quantization around 0.
            If False, uses asymmetric quantization with zero-point. Defaults to True.
        signed: Whether asymmetric quantization uses signed integers.
            Ignored if symmetric is True. Defaults to False.
        rounding: Rounding strategy instance. Defaults to NearestRounding().
        eps: Minimum threshold to prevent division by zero in scale computation.

    Returns:
        Tuple of (qx, scale, zero_point):
            - qx: Quantized integer tensor with identical shape and device as x.
            - scale: Column scale tensor of shape (..., 1, N).
            - zero_point: Column zero-point tensor of shape (..., 1, N), or None if symmetric.
    """
    if x.numel() == 0:
        return x.clone(), torch.empty(0, device=x.device, dtype=x.dtype), None

    if rounding is None:
        rounding = NearestRounding()

    # Dimension to reduce: for 1D, reduce dim 0; for >= 2D, reduce dim -2 (rows)
    reduce_dim = -2 if x.dim() >= 2 else 0

    if bits == 1 and symmetric:
        max_abs = torch.amax(torch.abs(x), dim=reduce_dim, keepdim=True)
        scale = torch.clamp(max_abs, min=eps)
        qx = torch.sign(x)
        qx[qx == 0] = 1.0
        return qx, scale, None

    qmin, qmax = _get_qmin_qmax(bits=bits, symmetric=symmetric, signed=signed)

    if symmetric:
        max_abs = torch.amax(torch.abs(x), dim=reduce_dim, keepdim=True)
        scale = torch.clamp(max_abs / qmax, min=eps)
        qx = x / scale
        qx = rounding.round(qx)
        qx = torch.clamp(qx, qmin, qmax)
        return qx, scale, None
    else:
        min_val = torch.amin(x, dim=reduce_dim, keepdim=True)
        max_val = torch.amax(x, dim=reduce_dim, keepdim=True)
        # Ensure 0.0 is represented in the dynamic range so that zero-point
        # maps cleanly without saturation clipping
        zero_t = torch.zeros(1, dtype=x.dtype, device=x.device)
        min_val = torch.minimum(min_val, zero_t)
        max_val = torch.maximum(max_val, zero_t)

        scale = torch.clamp((max_val - min_val) / float(qmax - qmin), min=eps)
        zp_real = float(qmin) - min_val / scale
        zero_point = torch.clamp(rounding.round(zp_real), qmin, qmax)

        qx = x / scale + zero_point
        qx = rounding.round(qx)
        qx = torch.clamp(qx, qmin, qmax)
        return qx, scale, zero_point


def per_column_dequantize(
    qx: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dequantize an integer tensor back to continuous floating-point values using per-column scales.

    Args:
        qx: Quantized tensor.
        scale: Per-column scale tensor (shape broadcastable to qx).
        zero_point: Optional per-column zero-point tensor (for asymmetric quantization).

    Returns:
        Dequantized floating-point tensor.
    """
    if qx.numel() == 0:
        return qx.clone()

    if zero_point is None:
        return qx * scale
    return (qx - zero_point) * scale


def per_column_fake_quantize(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    signed: Optional[bool] = None,
    rounding: Optional[RoundingStrategy] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Execute per-column quantization followed immediately by dequantization (fake quantization).

    Args:
        x: Input tensor.
        bits: Number of quantization bits.
        symmetric: Symmetric or asymmetric mode.
        signed: Signed integer mode for asymmetric quantization.
        rounding: Rounding strategy instance.
        eps: Minimum scale epsilon.

    Returns:
        Reconstructed floating-point tensor with simulated per-column quantization error.
    """
    qx, scale, zero_point = per_column_quantize(
        x=x,
        bits=bits,
        symmetric=symmetric,
        signed=signed,
        rounding=rounding,
        eps=eps,
    )
    return per_column_dequantize(qx=qx, scale=scale, zero_point=zero_point)


def per_axis_quantize(
    x: torch.Tensor,
    axis: Union[str, int] = "row",
    bits: int = 8,
    symmetric: bool = True,
    signed: Optional[bool] = None,
    rounding: Optional[RoundingStrategy] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Generalized per-axis quantization dispatcher.

    Allows specifying axis by name ('row', 'column', 'col') or integer dimension:
    - 'row', 'rows', or -2: Quantizes per-row (reduces over column axis).
    - 'column', 'col', 'columns', or -1: Quantizes per-column (reduces over row axis).

    Args:
        x: Input tensor.
        axis: Target axis ('row', 'column', or dimension index). Defaults to 'row'.
        bits: Quantization bit-width.
        symmetric: Symmetric or asymmetric mode.
        signed: Signed integer mode for asymmetric quantization.
        rounding: Rounding strategy instance.
        eps: Minimum scale epsilon.

    Returns:
        Tuple of (qx, scale, zero_point).
    """
    if isinstance(axis, str):
        normalized = axis.lower().strip()
        if normalized in ("row", "rows"):
            return per_row_quantize(
                x=x,
                bits=bits,
                symmetric=symmetric,
                signed=signed,
                rounding=rounding,
                eps=eps,
            )
        elif normalized in ("column", "col", "columns"):
            return per_column_quantize(
                x=x,
                bits=bits,
                symmetric=symmetric,
                signed=signed,
                rounding=rounding,
                eps=eps,
            )
        else:
            raise ValueError(
                f"Unrecognized axis string '{axis}'. Expected 'row' or 'column'."
            )

    # If axis is integer:
    # In 2D: axis=0 is row (preserve dim 0, reduce dim 1), axis=1 is column (preserve dim 1, reduce dim 0)
    dim_norm = axis if axis >= 0 else axis + x.dim()
    if x.dim() == 2:
        if dim_norm == 0:
            return per_row_quantize(
                x=x,
                bits=bits,
                symmetric=symmetric,
                signed=signed,
                rounding=rounding,
                eps=eps,
            )
        elif dim_norm == 1:
            return per_column_quantize(
                x=x,
                bits=bits,
                symmetric=symmetric,
                signed=signed,
                rounding=rounding,
                eps=eps,
            )
        else:
            raise ValueError(f"Axis index {axis} out of bounds for 2D tensor.")
    else:
        # General N-D tensor: preserve dim_norm, reduce all other dims
        reduce_dims = [d for d in range(x.dim()) if d != dim_norm]
        qmin, qmax = _get_qmin_qmax(bits=bits, symmetric=symmetric, signed=signed)
        if rounding is None:
            rounding = NearestRounding()

        if symmetric:
            max_abs = x.abs()
            for d in sorted(reduce_dims, reverse=True):
                max_abs = torch.amax(max_abs, dim=d, keepdim=True)
            scale = torch.clamp(max_abs / qmax, min=eps)
            qx = torch.clamp(rounding.round(x / scale), qmin, qmax)
            return qx, scale, None
        else:
            min_val = x
            max_val = x
            for d in sorted(reduce_dims, reverse=True):
                min_val = torch.amin(min_val, dim=d, keepdim=True)
                max_val = torch.amax(max_val, dim=d, keepdim=True)
            zero_t = torch.zeros(1, dtype=x.dtype, device=x.device)
            min_val = torch.minimum(min_val, zero_t)
            max_val = torch.maximum(max_val, zero_t)
            scale = torch.clamp((max_val - min_val) / float(qmax - qmin), min=eps)
            zp_real = float(qmin) - min_val / scale
            zero_point = torch.clamp(rounding.round(zp_real), qmin, qmax)
            qx = torch.clamp(rounding.round(x / scale + zero_point), qmin, qmax)
            return qx, scale, zero_point


def per_axis_dequantize(
    qx: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generalized per-axis dequantization.
    """
    if qx.numel() == 0:
        return qx.clone()
    if zero_point is None:
        return qx * scale
    return (qx - zero_point) * scale


def per_axis_fake_quantize(
    x: torch.Tensor,
    axis: Union[str, int] = "row",
    bits: int = 8,
    symmetric: bool = True,
    signed: Optional[bool] = None,
    rounding: Optional[RoundingStrategy] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized per-axis fake quantization.
    """
    qx, scale, zero_point = per_axis_quantize(
        x=x,
        axis=axis,
        bits=bits,
        symmetric=symmetric,
        signed=signed,
        rounding=rounding,
        eps=eps,
    )
    return per_axis_dequantize(qx=qx, scale=scale, zero_point=zero_point)


def vectorwise_quantized_matmul(
    activation: torch.Tensor,
    weight: torch.Tensor,
    bits: int = 8,
    rounding: Optional[RoundingStrategy] = None,
    eps: float = 1e-8,
    transpose_weight: bool = False,
) -> torch.Tensor:
    """
    Vector-wise INT8/INT4 Matrix Multiplication (LLM.int8() style).

    Computes GEMM C = A @ B using per-row activation quantization and per-column
    weight quantization:
    1. Quantize activation A (M, K) per-row: s_A in shape (M, 1).
    2. Quantize weight B (K, N) per-column: s_B in shape (1, N).
    3. Perform integer matrix multiplication: C_int = Q_A @ Q_B in shape (M, N).
    4. Rescale: C_hat = C_int * (s_A @ s_B) via outer-product scaling.

    Args:
        activation: Activation tensor of shape (M, K) or (..., M, K).
        weight: Weight matrix of shape (K, N), or (N, K) if transpose_weight is True.
        bits: Quantization bit-width (e.g., 8). Defaults to 8.
        rounding: Rounding strategy instance.
        eps: Minimum scale epsilon.
        transpose_weight: If True, transposes weight from (N, K) to (K, N). Defaults to False.

    Returns:
        Dequantized output tensor of shape (..., M, N).
    """
    w = weight.t() if transpose_weight else weight

    if activation.shape[-1] != w.shape[0]:
        raise ValueError(
            f"Inner matrix dimensions must match: activation {activation.shape} vs weight {w.shape}."
        )

    # 1. Per-row quantization of activations (preserves token / row axis)
    q_act, s_act, _ = per_row_quantize(
        x=activation,
        bits=bits,
        symmetric=True,
        rounding=rounding,
        eps=eps,
    )

    # 2. Per-column quantization of weight (preserves output column axis)
    q_w, s_w, _ = per_column_quantize(
        x=w,
        bits=bits,
        symmetric=True,
        rounding=rounding,
        eps=eps,
    )

    # 3. Integer matrix multiplication
    c_int = torch.matmul(q_act, q_w)

    # 4. Outer-product scale reconstruction
    scale_outer = s_act * s_w
    return c_int * scale_outer


class PerRowQuantizer(BaseQuantizer):
    """
    Object-oriented Per-Row Quantizer.

    Maintains per-row scaling factors and zero-points computed from the observed tensor.
    Inherits from BaseQuantizer.
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        signed: Optional[bool] = None,
        rounding: Optional[RoundingStrategy] = None,
        eps: float = 1e-8,
    ):
        super().__init__(bits=bits)
        self.symmetric = symmetric
        self.signed = signed
        self.rounding = rounding if rounding is not None else NearestRounding()
        self.eps = eps
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor per-row, caching scale and zero-point parameters.
        """
        qx, scale, zero_point = per_row_quantize(
            x=x,
            bits=self.bits,
            symmetric=self.symmetric,
            signed=self.signed,
            rounding=self.rounding,
            eps=self.eps,
        )
        self.scale = scale
        self.zero_point = zero_point
        return qx

    def dequantize(self, qx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Dequantize integer tensor using cached per-row parameters.
        """
        if self.scale is None:
            raise RuntimeError("Cannot dequantize before quantize() has been called.")
        if qx is None:
            raise ValueError("qx tensor must be provided for dequantization.")
        return per_row_dequantize(qx=qx, scale=self.scale, zero_point=self.zero_point)


class PerColumnQuantizer(BaseQuantizer):
    """
    Object-oriented Per-Column Quantizer.

    Maintains per-column scaling factors and zero-points computed from the observed tensor.
    Inherits from BaseQuantizer.
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        signed: Optional[bool] = None,
        rounding: Optional[RoundingStrategy] = None,
        eps: float = 1e-8,
    ):
        super().__init__(bits=bits)
        self.symmetric = symmetric
        self.signed = signed
        self.rounding = rounding if rounding is not None else NearestRounding()
        self.eps = eps
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor per-column, caching scale and zero-point parameters.
        """
        qx, scale, zero_point = per_column_quantize(
            x=x,
            bits=self.bits,
            symmetric=self.symmetric,
            signed=self.signed,
            rounding=self.rounding,
            eps=self.eps,
        )
        self.scale = scale
        self.zero_point = zero_point
        return qx

    def dequantize(self, qx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Dequantize integer tensor using cached per-column parameters.
        """
        if self.scale is None:
            raise RuntimeError("Cannot dequantize before quantize() has been called.")
        if qx is None:
            raise ValueError("qx tensor must be provided for dequantization.")
        return per_column_dequantize(
            qx=qx, scale=self.scale, zero_point=self.zero_point
        )


class PerAxisQuantizer(BaseQuantizer):
    """
    Object-oriented Per-Axis Quantizer.

    Unifies per-row, per-column, and arbitrary axis quantization.
    Inherits from BaseQuantizer.
    """

    def __init__(
        self,
        bits: int = 8,
        axis: Union[str, int] = "row",
        symmetric: bool = True,
        signed: Optional[bool] = None,
        rounding: Optional[RoundingStrategy] = None,
        eps: float = 1e-8,
    ):
        super().__init__(bits=bits)
        self.axis = axis
        self.symmetric = symmetric
        self.signed = signed
        self.rounding = rounding if rounding is not None else NearestRounding()
        self.eps = eps
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize tensor along configured axis, caching parameters.
        """
        qx, scale, zero_point = per_axis_quantize(
            x=x,
            axis=self.axis,
            bits=self.bits,
            symmetric=self.symmetric,
            signed=self.signed,
            rounding=self.rounding,
            eps=self.eps,
        )
        self.scale = scale
        self.zero_point = zero_point
        return qx

    def dequantize(self, qx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Dequantize tensor using cached parameters.
        """
        if self.scale is None:
            raise RuntimeError("Cannot dequantize before quantize() has been called.")
        if qx is None:
            raise ValueError("qx tensor must be provided for dequantization.")
        return per_axis_dequantize(qx=qx, scale=self.scale, zero_point=self.zero_point)


class PerRowLinear(nn.Module):
    """
    Drop-in replacement for PyTorch nn.Linear using per-row weight quantization.

    Weights are quantized row-wise (each output feature maintains an independent scale).
    Forward pass dequantizes weights dynamically to execute linear projections.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        symmetric: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.symmetric = symmetric

        self.quantizer = PerRowQuantizer(
            bits=bits,
            symmetric=symmetric,
        )

        # Buffers for quantized weights and per-row scales
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, in_features, device=device, dtype=torch.float32),
        )
        self.register_buffer(
            "scales",
            torch.ones(out_features, 1, device=device, dtype=dtype or torch.float32),
        )
        if not symmetric:
            self.register_buffer(
                "zero_points",
                torch.zeros(
                    out_features, 1, device=device, dtype=dtype or torch.float32
                ),
            )
        else:
            self.zero_points = None

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype or torch.float32)
            )
        else:
            self.register_parameter("bias", None)

    def set_weights(self, weight: torch.Tensor):
        """
        Quantize and register floating-point weight tensor in shape (out_features, in_features).
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

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 8,
        symmetric: bool = True,
    ) -> "PerRowLinear":
        """
        Construct a PerRowLinear module by quantizing an existing nn.Linear layer.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
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
        Reconstruct full-precision weights from per-row quantized buffer.
        """
        return per_row_dequantize(
            qx=self.qweight,
            scale=self.scales,
            zero_point=self.zero_points,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing linear projection with reconstructed weights.
        """
        weight_deq = self.get_dequantized_weight()
        return F.linear(x, weight_deq, self.bias)

    def memory_savings(self) -> dict:
        """
        Compute memory savings and compression ratio relative to FP32.
        """
        num_elements = self.in_features * self.out_features
        fp32_bytes = num_elements * 4.0

        qweight_bytes = (num_elements * self.bits) / 8.0
        scale_bytes = self.out_features * 4.0
        zp_bytes = self.out_features * 4.0 if not self.symmetric else 0.0
        bias_bytes = self.out_features * 4.0 if self.bias is not None else 0.0

        quantized_bytes = qweight_bytes + scale_bytes + zp_bytes + bias_bytes
        compression_ratio = fp32_bytes / max(quantized_bytes, 1.0)

        return {
            "num_elements": num_elements,
            "bits": self.bits,
            "granularity": "per-row",
            "fp32_bytes": fp32_bytes,
            "quantized_bytes": quantized_bytes,
            "compression_ratio": compression_ratio,
        }


class PerColumnLinear(nn.Module):
    """
    Drop-in replacement for PyTorch nn.Linear using per-column weight quantization.

    Weights are quantized column-wise (each input feature maintains an independent scale).
    Forward pass dequantizes weights dynamically to execute linear projections.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        symmetric: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.symmetric = symmetric

        self.quantizer = PerColumnQuantizer(
            bits=bits,
            symmetric=symmetric,
        )

        # Buffers for quantized weights and per-column scales
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, in_features, device=device, dtype=torch.float32),
        )
        self.register_buffer(
            "scales",
            torch.ones(1, in_features, device=device, dtype=dtype or torch.float32),
        )
        if not symmetric:
            self.register_buffer(
                "zero_points",
                torch.zeros(
                    1, in_features, device=device, dtype=dtype or torch.float32
                ),
            )
        else:
            self.zero_points = None

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype or torch.float32)
            )
        else:
            self.register_parameter("bias", None)

    def set_weights(self, weight: torch.Tensor):
        """
        Quantize and register floating-point weight tensor in shape (out_features, in_features).
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

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 8,
        symmetric: bool = True,
    ) -> "PerColumnLinear":
        """
        Construct a PerColumnLinear module by quantizing an existing nn.Linear layer.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
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
        Reconstruct full-precision weights from per-column quantized buffer.
        """
        return per_column_dequantize(
            qx=self.qweight,
            scale=self.scales,
            zero_point=self.zero_points,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing linear projection with reconstructed weights.
        """
        weight_deq = self.get_dequantized_weight()
        return F.linear(x, weight_deq, self.bias)

    def memory_savings(self) -> dict:
        """
        Compute memory savings and compression ratio relative to FP32.
        """
        num_elements = self.in_features * self.out_features
        fp32_bytes = num_elements * 4.0

        qweight_bytes = (num_elements * self.bits) / 8.0
        scale_bytes = self.in_features * 4.0
        zp_bytes = self.in_features * 4.0 if not self.symmetric else 0.0
        bias_bytes = self.out_features * 4.0 if self.bias is not None else 0.0

        quantized_bytes = qweight_bytes + scale_bytes + zp_bytes + bias_bytes
        compression_ratio = fp32_bytes / max(quantized_bytes, 1.0)

        return {
            "num_elements": num_elements,
            "bits": self.bits,
            "granularity": "per-column",
            "fp32_bytes": fp32_bytes,
            "quantized_bytes": quantized_bytes,
            "compression_ratio": compression_ratio,
        }
