"""
Mixed Precision Activations (8-bit Base + 16-bit Outliers) for LLMs.

Category 7: LLM-Specific Quantization (Milestone 7).

Overview & Motivation:
---------------------
In Large Language Models (LLMs) scaled beyond 6.7B parameters (e.g. OPT, BLOOM,
LLaMA, Falcon, Mistral), standard 8-bit quantization of activations causes
catastrophic degradation in perplexity and downstream task accuracy.

This phenomenon is driven by "emergent systematic activation outliers":
- A tiny subset of hidden feature dimensions (typically 6-10 channels out of 4096,
  or < 0.1% of feature coordinates) exhibits activation magnitudes 10x-100x larger
  than normal channels (e.g., |X| >= 6.0 while normal dimensions have |X| < 0.5).
- These outlier channels emerge systematically in specific features across sequence
  tokens and attention heads, carrying critical linguistic and syntactic information.
- When standard uniform INT8 quantization is applied to the activation tensor, the
  quantization scale factor s = max(|X|) / 127 is dictated entirely by these extreme
  outliers. As a result, non-outlier activations (representing >99.9% of all parameters)
  are rounded to zero or +/-1, destroying the model's representations (precision collapse).

The LLM.int8() Solution:
------------------------
Dettmers et al. (NeurIPS 2022) introduced mixed-precision matrix multiplication
with outlier decomposition:
1. Outlier Identification: Given activation X in R^(B x C) and threshold alpha (default 6.0),
   identify outlier channel coordinates:
       O = { c in {0, ..., C-1} | max_b |X_{b, c}| >= alpha }
       N = { c in {0, ..., C-1} | c not in O }
2. Decomposition:
   Split activation X into outlier features X_O in R^(B x |O|) and regular features
   X_N in R^(B x |N|). Split weight W into W_O in R^(D x |O|) and W_N in R^(D x |N|).
3. Dual-Path Execution:
   - Outlier Path (16-bit float / FP16 / BF16 / FP32):
       Y_O = X_O @ W_O^T
     Because |O| << C (typically <0.1%), this FP16 GEMM requires negligible FLOPs
     and zero quantization noise for extreme values.
   - Regular Path (8-bit quantized / INT8):
       Y_N = dequantize( quantize_int8(X_N) @ quantize_int8(W_N)^T )
     Because max |X_N| < alpha, INT8 quantization operates in its high-resolution
     regime without outlier-induced dynamic range stretching.
4. Linear Recombination:
       Y = Y_O + Y_N + bias
   By distributivity: X @ W^T = X_O @ W_O^T + X_N @ W_N^T.

References:
-----------
- Dettmers et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers
  at Scale." Advances in Neural Information Processing Systems (NeurIPS 2022).
  arXiv:2208.07339.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from inwhale.core.quantizer import BaseQuantizer


# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class MixedPrecisionState:
    """Container holding the decomposed mixed-precision activation state."""
    # 16-bit outlier activations (B x |O|) in outlier_dtype
    outliers: torch.Tensor
    # 1D LongTensor of channel indices corresponding to outliers (size |O|)
    outlier_indices: torch.Tensor
    # 8-bit quantized regular activations (B x |N|) in quant_dtype (torch.int8)
    quantized_regular: torch.Tensor
    # Quantization scales for regular activations (per-row or scalar)
    regular_scales: torch.Tensor
    # 1D LongTensor of channel indices corresponding to regular features (size |N|)
    regular_indices: torch.Tensor
    # Original tensor shape
    original_shape: torch.Size
    # Outlier threshold used during detection
    threshold: float
    # Floating point dtype of the input
    input_dtype: torch.dtype


# ============================================================================
# Outlier Detection & Decomposition Primitives
# ============================================================================

def find_outlier_channels(
    x: torch.Tensor,
    threshold: float = 6.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    Identify channel indices along `dim` where any element exceeds `threshold`.

    In transformer activations, systematic outliers emerge across specific
    feature dimensions (columns of X). If any token in the batch exhibits
    an activation magnitude |x| >= threshold in channel c, that channel is
    classified as an outlier dimension.

    Parameters:
    -----------
    x : torch.Tensor
        Input activation tensor of shape (*, C).
    threshold : float
        Magnitude threshold above which a channel is marked as an outlier
        (default: 6.0, per Dettmers et al. 2022).
    dim : int
        Channel/feature dimension along which outliers are identified (default: -1).

    Returns:
    --------
    torch.Tensor
        1D LongTensor containing sorted indices of outlier channels.
    """
    if threshold <= 0.0:
        raise ValueError(f"Outlier threshold must be strictly positive, got {threshold}.")

    # Normalize dimension index
    dim = dim % x.dim()

    # Move target channel dimension to the last axis if not already
    if dim != x.dim() - 1:
        x_perm = x.transpose(dim, -1)
    else:
        x_perm = x

    # Reshape to (M, C) where M = product of all non-channel dims
    num_channels = x_perm.shape[-1]
    x_flat = x_perm.reshape(-1, num_channels)

    # Compute maximum absolute activation per channel across all batch/token positions
    # Using float32 for stable comparison if tensor is half/bfloat16
    channel_max_abs = torch.max(torch.abs(x_flat), dim=0).values

    # Find channels where max magnitude >= threshold
    outlier_mask = channel_max_abs >= threshold
    outlier_indices = torch.nonzero(outlier_mask, as_tuple=False).squeeze(-1)

    return outlier_indices


def detect_outliers_by_quantile(
    x: torch.Tensor,
    quantile: float = 0.99,
    dim: int = -1,
) -> Tuple[torch.Tensor, float]:
    """
    Dynamically derive outlier threshold from channel maximum activation quantiles.

    Parameters:
    -----------
    x : torch.Tensor
        Input activation tensor of shape (*, C).
    quantile : float
        Channel-level quantile threshold (e.g. 0.99 for top 1% largest channels).
    dim : int
        Channel dimension.

    Returns:
    --------
    Tuple[torch.Tensor, float]
        (outlier_indices, computed_threshold)
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"Quantile must be in (0.0, 1.0), got {quantile}.")

    # Move target channel dimension to the last axis if not already
    dim = dim % x.dim()
    if dim != x.dim() - 1:
        x_perm = x.transpose(dim, -1)
    else:
        x_perm = x

    num_channels = x_perm.shape[-1]
    x_flat = x_perm.reshape(-1, num_channels)
    channel_max_abs = torch.max(torch.abs(x_flat), dim=0).values

    computed_threshold = float(torch.quantile(channel_max_abs.float(), quantile).item())
    computed_threshold = max(computed_threshold, 1e-4)

    outlier_indices = find_outlier_channels(x, threshold=computed_threshold, dim=dim)
    return outlier_indices, computed_threshold


def split_activation_channels(
    x: torch.Tensor,
    outlier_indices: torch.Tensor,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split activation tensor into outlier features and regular (non-outlier) features.

    Parameters:
    -----------
    x : torch.Tensor
        Activation tensor of shape (*, C).
    outlier_indices : torch.Tensor
        1D LongTensor of outlier channel indices.
    dim : int
        Channel dimension to split (default: -1).

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (x_outliers, x_regular, regular_indices)
        - x_outliers: Sliced tensor containing only outlier channels (*, |O|).
        - x_regular: Sliced tensor containing remaining channels (*, |N|).
        - regular_indices: 1D LongTensor of regular channel indices.
    """
    dim = dim % x.dim()
    num_channels = x.shape[dim]

    if outlier_indices.numel() == 0:
        # No outliers: all channels are regular
        regular_indices = torch.arange(num_channels, device=x.device, dtype=torch.long)
        empty_shape = list(x.shape)
        empty_shape[dim] = 0
        x_outliers = torch.empty(empty_shape, dtype=x.dtype, device=x.device)
        return x_outliers, x, regular_indices

    # Generate regular indices via boolean mask
    device = x.device
    mask = torch.ones(num_channels, dtype=torch.bool, device=device)
    mask[outlier_indices] = False
    regular_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)

    if regular_indices.numel() == 0:
        # All channels are outliers
        empty_shape = list(x.shape)
        empty_shape[dim] = 0
        x_regular = torch.empty(empty_shape, dtype=x.dtype, device=device)
        return x, x_regular, regular_indices

    # Vectorized indexing along dimension
    x_outliers = torch.index_select(x, dim, outlier_indices)
    x_regular = torch.index_select(x, dim, regular_indices)

    return x_outliers, x_regular, regular_indices


def split_weight_channels(
    w: torch.Tensor,
    outlier_indices: torch.Tensor,
    in_feature_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split 2D weight matrix (D x C) along input feature dimension C matching outlier channels.

    Parameters:
    -----------
    w : torch.Tensor
        Weight matrix of shape (out_features, in_features).
    outlier_indices : torch.Tensor
        1D LongTensor of outlier channel indices in [0, in_features - 1].
    in_feature_dim : int
        Dimension corresponding to input features (default: 1 for standard PyTorch Linear weight).

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        (w_outliers, w_regular)
        - w_outliers: Weight slice (out_features, |O|).
        - w_regular: Weight slice (out_features, |N|).
    """
    in_features = w.shape[in_feature_dim]

    if outlier_indices.numel() == 0:
        empty_shape = list(w.shape)
        empty_shape[in_feature_dim] = 0
        w_outliers = torch.empty(empty_shape, dtype=w.dtype, device=w.device)
        return w_outliers, w

    device = w.device
    mask = torch.ones(in_features, dtype=torch.bool, device=device)
    mask[outlier_indices] = False
    regular_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)

    if regular_indices.numel() == 0:
        empty_shape = list(w.shape)
        empty_shape[in_feature_dim] = 0
        w_regular = torch.empty(empty_shape, dtype=w.dtype, device=device)
        return w, w_regular

    w_outliers = torch.index_select(w, in_feature_dim, outlier_indices)
    w_regular = torch.index_select(w, in_feature_dim, regular_indices)

    return w_outliers, w_regular


# ============================================================================
# Vector-Wise & Tensor-Wise INT8 Quantization
# ============================================================================

def quantize_int8_vector_wise(
    x: torch.Tensor,
    qmax: int = 127,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vector-wise (row-wise) symmetric signed INT8 quantization.

    Computes independent scaling factor per row (token vector):
        s_i = max(|x_i|) / qmax
        q_i = clip(round(x_i / s_i), -qmax, qmax)

    Parameters:
    -----------
    x : torch.Tensor
        Input 2D tensor of shape (M, K).
    qmax : int
        Maximum symmetric integer representation (default: 127 for int8).
    eps : float
        Numerical epsilon preventing division by zero.

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        (q, scales) where q is torch.int8 of shape (M, K), and scales is shape (M, 1).
    """
    if x.numel() == 0:
        return torch.empty_like(x, dtype=torch.int8), torch.empty((x.shape[0], 1), dtype=x.dtype, device=x.device)

    # Compute row-wise absolute maximum: shape (M, 1)
    max_abs = torch.max(torch.abs(x), dim=-1, keepdim=True).values
    scales = torch.clamp(max_abs / float(qmax), min=eps)

    # Quantize and clamp to signed INT8 range [-127, 127]
    scaled = x / scales
    q = torch.clamp(torch.round(scaled), -qmax, qmax).to(torch.int8)

    return q, scales


def dequantize_int8_vector_wise(
    q: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize vector-wise INT8 tensor back to floating point:
        x_recon = q.to(dtype) * scales
    """
    return q.to(dtype) * scales


def quantize_int8_tensor_wise(
    x: torch.Tensor,
    qmax: int = 127,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tensor-wise symmetric signed INT8 quantization.

    Computes global scaling factor:
        s = max(|x|) / qmax
        q = clip(round(x / s), -qmax, qmax)
    """
    if x.numel() == 0:
        return torch.empty_like(x, dtype=torch.int8), torch.tensor(1.0, dtype=x.dtype, device=x.device)

    max_abs = torch.max(torch.abs(x))
    scale = torch.clamp(max_abs / float(qmax), min=eps)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)

    return q, scale


# ============================================================================
# Core Functional Mixed-Precision GEMM
# ============================================================================

def mixed_precision_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    threshold: float = 6.0,
    outlier_dtype: torch.dtype = torch.float16,
    vector_wise: bool = True,
) -> torch.Tensor:
    """
    Execute Mixed-Precision Matrix Multiplication with Outlier Decomposition (LLM.int8()).

    Computes Y = X @ W^T + bias by decomposing features into:
    1. Outlier channels (|X| >= threshold): computed in 16-bit floating point (outlier_dtype).
    2. Regular channels (|X| < threshold): quantized to signed INT8 and computed via INT8 GEMM.

    Parameters:
    -----------
    x : torch.Tensor
        Activation tensor of shape (..., in_features).
    w : torch.Tensor
        Weight matrix of shape (out_features, in_features).
    bias : Optional[torch.Tensor]
        Optional bias vector of shape (out_features,).
    threshold : float
        Outlier threshold for activation magnitudes (default: 6.0).
    outlier_dtype : torch.dtype
        Floating-point dtype for outlier path (default: torch.float16).
    vector_wise : bool
        If True, use row-wise vector quantization for regular activations and column-wise
        for weights (default: True, per LLM.int8()). If False, use per-tensor scaling.

    Returns:
    --------
    torch.Tensor
        Output activation tensor of shape (..., out_features) in input's dtype.
    """
    orig_shape = x.shape
    in_features = w.shape[1]
    out_features = w.shape[0]

    if orig_shape[-1] != in_features:
        raise ValueError(
            f"Shape mismatch: x has in_features={orig_shape[-1]}, but w has in_features={in_features}."
        )

    # Flatten leading batch/sequence dimensions: (M, C)
    x_2d = x.reshape(-1, in_features)
    input_dtype = x.dtype
    device = x.device

    # 1. Identify outlier channels across all tokens
    outlier_indices = find_outlier_channels(x_2d, threshold=threshold, dim=-1)
    num_outliers = outlier_indices.numel()

    # 2. Case A: No outliers found (|O| = 0) -> Pure INT8 regular path
    if num_outliers == 0:
        y_2d = _regular_int8_gemm(x_2d, w, vector_wise=vector_wise, out_dtype=input_dtype)
        if bias is not None:
            y_2d = y_2d + bias
        out_shape = list(orig_shape[:-1]) + [out_features]
        return y_2d.reshape(out_shape)

    # 3. Case B: All channels are outliers (|O| = C) -> Pure 16-bit floating path
    if num_outliers == in_features:
        x_out = x_2d.to(outlier_dtype)
        w_out = w.to(outlier_dtype)
        y_2d = F.linear(x_out, w_out, bias=bias.to(outlier_dtype) if bias is not None else None)
        out_shape = list(orig_shape[:-1]) + [out_features]
        return y_2d.to(input_dtype).reshape(out_shape)

    # 4. Case C: Hybrid Mixed Precision Decomposition (0 < |O| < C)
    x_outliers, x_regular, regular_indices = split_activation_channels(x_2d, outlier_indices, dim=-1)
    w_outliers, w_regular = split_weight_channels(w, outlier_indices, in_feature_dim=1)

    # Path 1: 16-bit Float Outlier GEMM (X_O @ W_O^T)
    x_out_cast = x_outliers.to(outlier_dtype)
    w_out_cast = w_outliers.to(outlier_dtype)
    y_outliers = torch.matmul(x_out_cast, w_out_cast.t()).to(input_dtype)

    # Path 2: INT8 Quantized Regular GEMM (X_N @ W_N^T)
    y_regular = _regular_int8_gemm(x_regular, w_regular, vector_wise=vector_wise, out_dtype=input_dtype)

    # Combine both paths: Y = Y_O + Y_N + bias
    y_total = y_outliers + y_regular
    if bias is not None:
        y_total = y_total + bias

    out_shape = list(orig_shape[:-1]) + [out_features]
    return y_total.reshape(out_shape)


def _regular_int8_gemm(
    x_reg: torch.Tensor,
    w_reg: torch.Tensor,
    vector_wise: bool = True,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Internal helper executing INT8 matrix multiplication for non-outlier activations and weights.
    """
    if x_reg.numel() == 0 or w_reg.numel() == 0:
        return torch.zeros((x_reg.shape[0], w_reg.shape[0]), dtype=out_dtype, device=x_reg.device)

    if vector_wise:
        # Row-wise scaling for X: (M, 1)
        qx, sx = quantize_int8_vector_wise(x_reg)
        # Row-wise scaling for W (which corresponds to output channel): (D, 1)
        qw, sw = quantize_int8_vector_wise(w_reg)

        # Integer GEMM accumulated in float32 for seamless cross-platform CUDA/CPU compatibility
        acc = torch.matmul(qx.to(torch.float32), qw.to(torch.float32).t()).to(out_dtype)

        # Dequantize: acc * (sx @ sw^T)
        scale_mat = torch.matmul(sx.to(out_dtype), sw.to(out_dtype).t())
        return acc * scale_mat
    else:
        # Tensor-wise scaling
        qx, sx = quantize_int8_tensor_wise(x_reg)
        qw, sw = quantize_int8_tensor_wise(w_reg)

        acc = torch.matmul(qx.to(torch.float32), qw.to(torch.float32).t()).to(out_dtype)
        scale_val = (sx * sw).to(out_dtype)
        return acc * scale_val


# ============================================================================
# Autograd Function with Straight-Through Estimator (STE)
# ============================================================================

class _MixedPrecisionMatmulFunction(torch.autograd.Function):
    """
    Autograd function implementing Mixed-Precision Matrix Multiplication with STE.

    During backward:
    - Outlier features receive exact floating-point gradients.
    - Regular features receive straight-through gradients passing through INT8 quantization:
      dL/dX_regular = dL/dY @ W_regular.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: Optional[torch.Tensor],
        threshold: float,
        outlier_dtype: torch.dtype,
        vector_wise: bool,
    ) -> torch.Tensor:
        orig_shape = x.shape
        in_features = w.shape[1]
        x_2d = x.reshape(-1, in_features)

        # Detect outliers
        outlier_indices = find_outlier_channels(x_2d, threshold=threshold, dim=-1)

        # Forward calculation
        y_2d = mixed_precision_matmul(
            x_2d,
            w,
            bias=bias,
            threshold=threshold,
            outlier_dtype=outlier_dtype,
            vector_wise=vector_wise,
        )

        # Save for backward pass
        ctx.save_for_backward(x_2d, w, bias, outlier_indices)
        ctx.threshold = threshold
        ctx.vector_wise = vector_wise
        ctx.orig_shape = orig_shape

        out_shape = list(orig_shape[:-1]) + [w.shape[0]]
        return y_2d.reshape(out_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, w, bias, outlier_indices = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        out_features = w.shape[0]

        grad_output_2d = grad_output.reshape(-1, out_features)

        grad_x = None
        grad_w = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            # Gradient w.r.t input activation X:
            # dL/dX = dL/dY @ W (Straight-Through Estimator for regular path, exact for outliers)
            grad_x_2d = torch.matmul(grad_output_2d, w)
            grad_x = grad_x_2d.reshape(orig_shape)

        if ctx.needs_input_grad[1]:
            # Gradient w.r.t weight matrix W:
            # dL/dW = (dL/dY)^T @ X
            grad_w = torch.matmul(grad_output_2d.t(), x_2d)

        if bias is not None and ctx.needs_input_grad[2]:
            # Gradient w.r.t bias:
            grad_bias = torch.sum(grad_output_2d, dim=0)

        return grad_x, grad_w, grad_bias, None, None, None


def mixed_precision_matmul_ste(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    threshold: float = 6.0,
    outlier_dtype: torch.dtype = torch.float16,
    vector_wise: bool = True,
) -> torch.Tensor:
    """
    Functional entry point for Mixed-Precision Matmul supporting STE backpropagation.
    """
    return _MixedPrecisionMatmulFunction.apply(
        x, w, bias, threshold, outlier_dtype, vector_wise
    )


# ============================================================================
# Object-Oriented Quantizer
# ============================================================================

class MixedPrecisionActivationQuantizer(BaseQuantizer):
    """
    Object-oriented quantizer for Mixed-Precision Activations (8-bit base + 16-bit outliers).

    Inherits from `BaseQuantizer`. Implements decomposition, quantization, dequantization,
    and outlier diagnostics.

    Parameters:
    -----------
    bits : int
        Bit-width for regular activations (default: 8).
    threshold : float
        Outlier activation magnitude threshold (default: 6.0).
    outlier_dtype : torch.dtype
        Data type for storing and calculating outliers (default: torch.float16).
    vector_wise : bool
        Whether to use vector-wise (per-row) scaling for regular features (default: True).
    """

    def __init__(
        self,
        bits: int = 8,
        threshold: float = 6.0,
        outlier_dtype: torch.dtype = torch.float16,
        vector_wise: bool = True,
    ):
        super().__init__(bits)
        if threshold <= 0.0:
            raise ValueError(f"Threshold must be positive, got {threshold}.")
        self.threshold = threshold
        self.outlier_dtype = outlier_dtype
        self.vector_wise = vector_wise
        self.qmin = -(1 << (bits - 1)) + 1  # -127 for bits=8
        self.qmax = (1 << (bits - 1)) - 1   # +127 for bits=8

    def quantize(self, x: torch.Tensor) -> MixedPrecisionState:
        """
        Quantize input tensor into MixedPrecisionState containing 16-bit outliers and 8-bit regular activations.

        Parameters:
        -----------
        x : torch.Tensor
            Input activation tensor of shape (*, C).

        Returns:
        --------
        MixedPrecisionState
            Decomposed state object.
        """
        orig_shape = x.shape
        num_channels = orig_shape[-1]
        x_2d = x.reshape(-1, num_channels)

        outlier_indices = find_outlier_channels(x_2d, threshold=self.threshold, dim=-1)
        x_outliers, x_regular, regular_indices = split_activation_channels(x_2d, outlier_indices, dim=-1)

        # Store outliers in 16-bit float
        outliers_cast = x_outliers.to(self.outlier_dtype)

        # Quantize regular activations to signed INT8
        if self.vector_wise:
            q_regular, scales = quantize_int8_vector_wise(x_regular, qmax=self.qmax)
        else:
            q_regular, scales = quantize_int8_tensor_wise(x_regular, qmax=self.qmax)

        return MixedPrecisionState(
            outliers=outliers_cast,
            outlier_indices=outlier_indices,
            quantized_regular=q_regular,
            regular_scales=scales,
            regular_indices=regular_indices,
            original_shape=orig_shape,
            threshold=self.threshold,
            input_dtype=x.dtype,
        )

    def dequantize(self, state: MixedPrecisionState) -> torch.Tensor:
        """
        Reconstruct full activation tensor from MixedPrecisionState.

        Parameters:
        -----------
        state : MixedPrecisionState
            Quantized state.

        Returns:
        --------
        torch.Tensor
            Reconstructed activation tensor of original shape (*, C).
        """
        orig_shape = state.original_shape
        num_channels = orig_shape[-1]
        total_rows = math.prod(orig_shape[:-1]) if len(orig_shape) > 1 else 1
        device = state.quantized_regular.device
        target_dtype = state.input_dtype

        # Allocate reconstruction buffer
        recon_2d = torch.zeros((total_rows, num_channels), dtype=target_dtype, device=device)

        # Dequantize regular activations
        if state.regular_indices.numel() > 0:
            if self.vector_wise:
                x_reg = dequantize_int8_vector_wise(state.quantized_regular, state.regular_scales, dtype=target_dtype)
            else:
                x_reg = state.quantized_regular.to(target_dtype) * state.regular_scales.to(target_dtype)
            recon_2d[:, state.regular_indices] = x_reg

        # Insert 16-bit outliers
        if state.outlier_indices.numel() > 0:
            recon_2d[:, state.outlier_indices] = state.outliers.to(target_dtype)

        return recon_2d.reshape(orig_shape)

    def matmul(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute mixed-precision matrix multiplication Y = X @ W^T + bias.
        """
        return mixed_precision_matmul(
            x,
            w,
            bias=bias,
            threshold=self.threshold,
            outlier_dtype=self.outlier_dtype,
            vector_wise=self.vector_wise,
        )

    def analyze_outliers(self, x: torch.Tensor) -> Dict[str, Union[int, float, List[int]]]:
        """
        Diagnostic summary analyzing activation outlier distribution across channels.

        Returns:
        --------
        dict containing:
            - 'total_channels': Total number of features (C).
            - 'num_outliers': Number of outlier channels (|O|).
            - 'outlier_ratio': Percentage of channels that are outliers.
            - 'outlier_indices': List of outlier channel indices.
            - 'max_activation': Peak activation magnitude across tensor.
            - 'mean_activation': Mean activation magnitude across tensor.
            - 'threshold': Outlier threshold used.
        """
        orig_shape = x.shape
        num_channels = orig_shape[-1]
        x_2d = x.reshape(-1, num_channels)

        outlier_indices = find_outlier_channels(x_2d, threshold=self.threshold, dim=-1)
        num_outliers = int(outlier_indices.numel())
        outlier_ratio = float(num_outliers / num_channels) if num_channels > 0 else 0.0

        max_act = float(torch.max(torch.abs(x_2d)).item()) if x_2d.numel() > 0 else 0.0
        mean_act = float(torch.mean(torch.abs(x_2d.float())).item()) if x_2d.numel() > 0 else 0.0

        return {
            "total_channels": num_channels,
            "num_outliers": num_outliers,
            "outlier_ratio": outlier_ratio,
            "outlier_indices": outlier_indices.cpu().tolist(),
            "max_activation": max_act,
            "mean_activation": mean_act,
            "threshold": self.threshold,
        }


# ============================================================================
# Drop-in PyTorch Neural Network Module
# ============================================================================

class MixedPrecisionLinear(nn.Module):
    """
    Drop-in replacement for `nn.Linear` utilizing Mixed-Precision Activations (LLM.int8()).

    During forward inference:
    - Inspects incoming activations for systematic outlier channels (|X| >= threshold).
    - Routes outlier channels through 16-bit floating-point GEMM.
    - Routes regular channels through 8-bit integer GEMM.
    - Achieves high compression and acceleration without precision degradation.

    Parameters:
    -----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool
        If set to False, layer will not learn an additive bias (default: True).
    threshold : float
        Activation magnitude outlier cutoff (default: 6.0).
    outlier_dtype : torch.dtype
        Data type for outlier path calculations (default: torch.float16).
    vector_wise : bool
        Whether to use row-wise and column-wise vector scaling for regular path (default: True).
    static_outlier_indices : Optional[torch.Tensor]
        Optional pre-calibrated 1D tensor of known systematic outlier channel indices.
        If provided, skips dynamic outlier detection during forward passes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 6.0,
        outlier_dtype: torch.dtype = torch.float16,
        vector_wise: bool = True,
        static_outlier_indices: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.outlier_dtype = outlier_dtype
        self.vector_wise = vector_wise

        # Initialize full precision weight parameter
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Static outlier channel cache (optional)
        if static_outlier_indices is not None:
            self.register_buffer("static_outliers", static_outlier_indices.to(torch.long))
        else:
            self.register_buffer("static_outliers", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights matching standard nn.Linear initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_static_outliers(self, outlier_indices: torch.Tensor):
        """Cache calibrated systematic outlier channel indices."""
        self.static_outliers = outlier_indices.to(device=self.weight.device, dtype=torch.long)

    def clear_static_outliers(self):
        """Clear cached static outlier indices and revert to dynamic detection."""
        self.static_outliers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mixed-precision activation decomposition.
        """
        if self.training:
            # During training, use STE to allow gradient flow
            return mixed_precision_matmul_ste(
                x,
                self.weight,
                bias=self.bias,
                threshold=self.threshold,
                outlier_dtype=self.outlier_dtype,
                vector_wise=self.vector_wise,
            )

        if self.static_outliers is not None:
            # Calibrated static outlier path
            orig_shape = x.shape
            x_2d = x.reshape(-1, self.in_features)
            x_out, x_reg, _ = split_activation_channels(x_2d, self.static_outliers, dim=-1)
            w_out, w_reg = split_weight_channels(self.weight, self.static_outliers, in_feature_dim=1)

            # Outlier GEMM
            y_out = torch.matmul(x_out.to(self.outlier_dtype), w_out.to(self.outlier_dtype).t()).to(x.dtype)
            # Regular INT8 GEMM
            y_reg = _regular_int8_gemm(x_reg, w_reg, vector_wise=self.vector_wise, out_dtype=x.dtype)

            y_total = y_out + y_reg
            if self.bias is not None:
                y_total = y_total + self.bias
            out_shape = list(orig_shape[:-1]) + [self.out_features]
            return y_total.reshape(out_shape)

        return mixed_precision_matmul(
            x,
            self.weight,
            bias=self.bias,
            threshold=self.threshold,
            outlier_dtype=self.outlier_dtype,
            vector_wise=self.vector_wise,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        threshold: float = 6.0,
        outlier_dtype: torch.dtype = torch.float16,
        vector_wise: bool = True,
        static_outlier_indices: Optional[torch.Tensor] = None,
    ) -> "MixedPrecisionLinear":
        """
        Construct MixedPrecisionLinear directly from an existing nn.Linear module.

        Parameters:
        -----------
        linear : nn.Linear
            Trained PyTorch linear layer.
        threshold : float
            Activation outlier threshold.
        outlier_dtype : torch.dtype
            Precision for outlier calculations.
        vector_wise : bool
            Whether to use vector-wise INT8 quantization.
        static_outlier_indices : Optional[torch.Tensor]
            Pre-calibrated outlier indices.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            threshold=threshold,
            outlier_dtype=outlier_dtype,
            vector_wise=vector_wise,
            static_outlier_indices=static_outlier_indices,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def memory_stats(self) -> Dict[str, Union[int, float]]:
        """
        Compute memory footprint of weights and layer parameters.
        """
        weight_bytes = self.weight.numel() * self.weight.element_size()
        bias_bytes = self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        total_bytes = weight_bytes + bias_bytes

        # Theoretical INT8 quantized footprint: 1 byte per weight element + 4 bytes scale per row
        int8_weight_bytes = self.weight.numel() * 1 + self.out_features * 4
        compression_ratio = weight_bytes / max(int8_weight_bytes, 1)

        return {
            "total_bytes": total_bytes,
            "weight_bytes": weight_bytes,
            "bias_bytes": bias_bytes,
            "theoretical_int8_bytes": int8_weight_bytes + bias_bytes,
            "compression_ratio": compression_ratio,
        }

    def extra_repr(self) -> str:
        s = f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        s += f", threshold={self.threshold}, outlier_dtype={self.outlier_dtype}, vector_wise={self.vector_wise}"
        if self.static_outliers is not None:
            s += f", static_outliers={self.static_outliers.numel()}"
        return s


# ============================================================================
# Benchmarking & Diagnostic Comparison Tool
# ============================================================================

def compare_mixed_vs_uniform_8bit(
    x: torch.Tensor,
    w: torch.Tensor,
    threshold: float = 6.0,
    vector_wise: bool = True,
) -> Dict[str, Union[float, int]]:
    """
    Compare numerical fidelity between Full Precision (FP32), Standard INT8,
    and Mixed-Precision Activations (8-bit + 16-bit Outliers).

    Demonstrates the catastrophic precision collapse of uniform INT8 in the
    presence of activation outliers and the near-lossless fidelity recovery
    provided by mixed-precision outlier decomposition.

    Metrics computed:
    - Signal-to-Noise Ratio (SNR in dB): 10 * log10( ||Y_fp32||^2 / ||Y_approx - Y_fp32||^2 )
    - Cosine Similarity: dot(Y_approx, Y_fp32) / (||Y_approx|| * ||Y_fp32||)
    - Mean Absolute Error (MAE)
    - Max Absolute Error
    """
    with torch.no_grad():
        # Full precision reference output (FP32)
        x_fp32 = x.float()
        w_fp32 = w.float()
        y_ref = F.linear(x_fp32, w_fp32)

        # Method 1: Standard Pure INT8 (no outlier isolation)
        x_reg_q, x_reg_s = quantize_int8_vector_wise(x_fp32) if vector_wise else quantize_int8_tensor_wise(x_fp32)
        w_reg_q, w_reg_s = quantize_int8_vector_wise(w_fp32) if vector_wise else quantize_int8_tensor_wise(w_fp32)
        if vector_wise:
            acc_pure = torch.matmul(x_reg_q.float(), w_reg_q.float().t())
            scale_pure = torch.matmul(x_reg_s.float(), w_reg_s.float().t())
            y_pure_int8 = acc_pure * scale_pure
        else:
            acc_pure = torch.matmul(x_reg_q.float(), w_reg_q.float().t())
            y_pure_int8 = acc_pure * (x_reg_s * w_reg_s).float()

        # Method 2: Mixed-Precision (8-bit + 16-bit outliers)
        y_mixed = mixed_precision_matmul(
            x_fp32,
            w_fp32,
            threshold=threshold,
            outlier_dtype=torch.float16,
            vector_wise=vector_wise,
        )

        def calc_metrics(approx: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float, float, float]:
            diff = approx.float() - ref.float()
            signal_power = torch.sum(ref.float() ** 2).item()
            noise_power = torch.sum(diff ** 2).item()
            snr = 10.0 * math.log10(max(signal_power, 1e-12) / max(noise_power, 1e-12))

            cos_sim = F.cosine_similarity(approx.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
            mae = torch.mean(torch.abs(diff)).item()
            max_err = torch.max(torch.abs(diff)).item()
            return snr, cos_sim, mae, max_err

        snr_pure, cos_pure, mae_pure, max_pure = calc_metrics(y_pure_int8, y_ref)
        snr_mix, cos_mix, mae_mix, max_mix = calc_metrics(y_mixed, y_ref)

        outlier_indices = find_outlier_channels(x_fp32.reshape(-1, x_fp32.shape[-1]), threshold=threshold)

        return {
            "num_outliers": int(outlier_indices.numel()),
            "total_channels": int(x.shape[-1]),
            "threshold": threshold,
            "pure_int8_snr_db": snr_pure,
            "pure_int8_cosine_sim": cos_pure,
            "pure_int8_mae": mae_pure,
            "pure_int8_max_err": max_pure,
            "mixed_precision_snr_db": snr_mix,
            "mixed_precision_cosine_sim": cos_mix,
            "mixed_precision_mae": mae_mix,
            "mixed_precision_max_err": max_mix,
            "snr_improvement_db": snr_mix - snr_pure,
        }
