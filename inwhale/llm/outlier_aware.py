"""
Outlier-Aware Activation Quantization (SmoothQuant & Channel Migration) for LLMs.

Category 7: LLM-Specific Quantization (Milestone 7).

Overview & Motivation:
---------------------
In Large Language Models (LLMs) scaled beyond 6.7B parameters (e.g. OPT, BLOOM,
LLaMA, Falcon, Mistral, Mixtral), activations exhibit extreme channel-wise kurtosis:
- A tiny fraction of feature channels (typically 0.1% - 0.2%) systematically produce
  activation magnitudes 10x - 100x larger than normal channels (e.g., |X| >= 50.0
  while normal dimensions have |X| < 1.0).
- These emergent outlier channels are consistent across sequence tokens and carry
  critical semantic information.
- In standard uniform 8-bit (INT8) quantization, the dynamic range scale factor:
      s_X = max(|X|) / 127
  is dominated entirely by these extreme outliers. Consequently, non-outlier channels
  (representing >99.8% of all features) are rounded to 0 or +/-1, destroying the
  model's representations (catastrophic precision collapse).
- Unlike per-channel weight quantization, per-channel activation quantization
  (column-wise scaling of X in R^(B x C_in)) cannot be accelerated directly by
  standard hardware matrix multiplication cores (e.g. Tensor Cores, INT8 DP4A/WMMA)
  because the inner reduction dimension C_in requires a single shared scale between
  activation and weight elements.

The SmoothQuant Solution (Xiao et al., ICML 2023):
--------------------------------------------------
SmoothQuant addresses this fundamental challenge by observing that weights W
are much easier to quantize than activations X (weights have uniform Gaussian-like
distributions without extreme channel outliers).

SmoothQuant mathematically migrates the quantization difficulty from activations
to weights using an invertible diagonal channel-scaling transform:
    Y = X @ W^T = (X @ diag(s)^(-1)) @ (diag(s) @ W^T) = X_hat @ W_hat^T

where s in R^(C_in) (s_j > 0) is the per-channel smoothing scale vector.

Element-wise:
    X_hat_{b, j} = X_{b, j} / s_j
    W_hat_{i, j} = W_{i, j} * s_j

Optimal Migration Scale (SmoothQuant Scale Formula):
---------------------------------------------------
To optimally balance quantization difficulty between activations and weights,
the per-channel scale s_j is calculated as:
    s_j = ( max_b |X_{b, j}| )^alpha / ( max_i |W_{i, j}| )^(1 - alpha)

where:
- max_b |X_{b, j}| is the maximum absolute activation magnitude in channel j
  (estimated via calibration data or dynamic observation).
- max_i |W_{i, j}| is the maximum absolute weight magnitude across all output rows
  in input channel j.
- alpha in [0, 1] is the migration strength hyperparameter:
    * alpha = 0.5 (default): geometric mean balance, optimal for most LLM architectures.
    * alpha = 1.0: total migration (activations are completely smoothed; weights absorb
      all channel dynamic range variance).
    * alpha = 0.0: no migration (weights are divided by their channel maximums).

Zero-Overhead Inference via LayerNorm Folding:
----------------------------------------------
In Transformer architectures, Linear layers (Q, K, V projections and FFN Up/Gate)
immediately follow a normalization layer (LayerNorm or RMSNorm):
    LN(z) = ((z - mu) / sigma) * gamma + beta

The smoothing transform diag(s)^(-1) can be folded directly into the LayerNorm parameters:
    gamma_hat_j = gamma_j / s_j
    beta_hat_j  = beta_j  / s_j

and pre-multiplied into the linear weight:
    W_hat_{i, j} = W_{i, j} * s_j

Thus, at inference time, LayerNorm directly outputs smoothed activations X_hat,
and the linear layer executes standard high-speed INT8 W8A8 GEMM with ZERO
additional runtime FLOPs or latency overhead!

References:
-----------
- Xiao et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization
  for Large Language Models." International Conference on Machine Learning (ICML 2023).
  arXiv:2211.10438.
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
class SmoothQuantState:
    """Container holding the state of smoothed and quantized activations."""
    # Smoothed activations X_hat = X / s
    smoothed_x: torch.Tensor
    # Quantized integer activations (torch.int8)
    quantized_x: torch.Tensor
    # Quantization scale for activations (per-token or scalar)
    act_scale: torch.Tensor
    # Per-channel smoothing migration scales s in R^(C_in)
    migration_scale: torch.Tensor
    # Original tensor shape
    original_shape: torch.Size
    # Migration strength alpha
    alpha: float
    # Floating-point dtype of the input
    input_dtype: torch.dtype


@dataclass
class OutlierDiagnostics:
    """Container holding activation distribution and outlier diagnostic metrics."""
    # Number of channels analyzed
    num_channels: int
    # 1D tensor of per-channel peak absolute values max_b |X_{b, j}|
    channel_peaks: torch.Tensor
    # Global maximum absolute activation value
    global_max: float
    # Global mean absolute activation value
    global_mean: float
    # Peak-to-Average Ratio (PAR = global_max / global_mean)
    par: float
    # Kurtosis of per-channel peak distribution
    kurtosis: float
    # Indices of outlier channels exceeding threshold
    outlier_channel_indices: torch.Tensor
    # Fraction of channels classified as outliers
    outlier_ratio: float


# ============================================================================
# Core Mathematical Primitives: Scale Computation & Migration
# ============================================================================

def compute_smoothquant_scales(
    act_scales: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute optimal per-channel smoothing scale vector s for SmoothQuant.

    Mathematical formulation:
        s_j = ( max_b |X_{b, j}| )^alpha / ( max_i |W_{i, j}| )^(1 - alpha)
        s_j = clamp(s_j, min=eps, max=1.0/eps)

    Parameters:
    -----------
    act_scales : torch.Tensor
        1D tensor of shape (C_in,) representing per-channel maximum absolute
        activation magnitudes (e.g. from calibration data or batch statistics).
    weight : torch.Tensor
        2D weight tensor of shape (C_out, C_in) or 1D weight channel max of shape (C_in,).
    alpha : float
        Migration strength parameter in [0.0, 1.0].
        0.5 balances activation and weight quantization difficulty equally.
    eps : float
        Numerical stability constant preventing division by zero or extreme scales.

    Returns:
    --------
    torch.Tensor
        1D smoothing scale tensor s of shape (C_in,) with positive values.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

    if act_scales.ndim != 1:
        raise ValueError(f"act_scales must be 1D with shape (C_in,), got shape {act_scales.shape}")

    c_in = act_scales.shape[0]

    if weight.ndim == 2:
        if weight.shape[1] != c_in:
            raise ValueError(
                f"Weight matrix second dimension ({weight.shape[1]}) must match "
                f"act_scales dimension ({c_in})"
            )
        # Compute maximum absolute weight along output dimension (dim 0) -> shape (C_in,)
        weight_scales = torch.max(torch.abs(weight), dim=0).values
    elif weight.ndim == 1:
        if weight.shape[0] != c_in:
            raise ValueError(
                f"Weight scale dimension ({weight.shape[0]}) must match "
                f"act_scales dimension ({c_in})"
            )
        weight_scales = torch.abs(weight)
    else:
        raise ValueError(f"weight must be 1D or 2D tensor, got shape {weight.shape}")

    # Ensure device and dtype consistency
    device = act_scales.device
    dtype = act_scales.dtype
    weight_scales = weight_scales.to(device=device, dtype=dtype)

    # Clamping for numerical stability
    act_clamped = torch.clamp(act_scales, min=eps)
    weight_clamped = torch.clamp(weight_scales, min=eps)

    # Compute numerator: act_scales^alpha
    if alpha == 0.0:
        act_term = torch.ones_like(act_clamped)
    elif alpha == 1.0:
        act_term = act_clamped
    elif alpha == 0.5:
        act_term = torch.sqrt(act_clamped)
    else:
        act_term = torch.pow(act_clamped, alpha)

    # Compute denominator: weight_scales^(1 - alpha)
    beta = 1.0 - alpha
    if beta == 0.0:
        weight_term = torch.ones_like(weight_clamped)
    elif beta == 1.0:
        weight_term = weight_clamped
    elif beta == 0.5:
        weight_term = torch.sqrt(weight_clamped)
    else:
        weight_term = torch.pow(weight_clamped, beta)

    # Smoothing scale s_j
    scales = act_term / torch.clamp(weight_term, min=eps)

    # Guard against zero, nan, or extreme scales
    max_scale = 1.0 / eps
    scales = torch.clamp(scales, min=eps, max=max_scale)

    return scales


def apply_activation_smoothing(
    x: torch.Tensor,
    smoothing_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Apply per-channel activation smoothing: X_hat = X / s.

    Parameters:
    -----------
    x : torch.Tensor
        Activation tensor of shape (..., C_in).
    smoothing_scale : torch.Tensor
        1D smoothing scale vector s of shape (C_in,).

    Returns:
    --------
    torch.Tensor
        Smoothed activation tensor X_hat with identical shape and dtype as x.
    """
    c_in = x.shape[-1]
    if smoothing_scale.shape[-1] != c_in:
        raise ValueError(
            f"smoothing_scale shape {smoothing_scale.shape} does not match "
            f"activation channel dimension {c_in}"
        )

    # Ensure scale is on matching device and dtype
    scale = smoothing_scale.to(device=x.device, dtype=x.dtype)
    return x / scale


def apply_weight_smoothing(
    weight: torch.Tensor,
    smoothing_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Apply per-channel weight smoothing: W_hat = W * s.

    For weight matrix W of shape (C_out, C_in):
        W_hat_{i, j} = W_{i, j} * s_j

    Parameters:
    -----------
    weight : torch.Tensor
        Linear layer weight matrix of shape (C_out, C_in).
    smoothing_scale : torch.Tensor
        1D smoothing scale vector s of shape (C_in,).

    Returns:
    --------
    torch.Tensor
        Smoothed weight matrix W_hat of shape (C_out, C_in).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be a 2D matrix (C_out, C_in), got shape {weight.shape}")

    c_in = weight.shape[1]
    if smoothing_scale.shape[0] != c_in:
        raise ValueError(
            f"smoothing_scale length ({smoothing_scale.shape[0]}) must match "
            f"weight input channels ({c_in})"
        )

    scale = smoothing_scale.to(device=weight.device, dtype=weight.dtype)
    # Broadcast scale along output rows (dim 0): (C_out, C_in) * (1, C_in)
    return weight * scale.unsqueeze(0)


def fold_smoothing_into_layernorm(
    layernorm: Union[nn.LayerNorm, nn.Module],
    smoothing_scale: torch.Tensor,
) -> None:
    """
    Fold per-channel smoothing scale s into LayerNorm / RMSNorm parameters in-place.

    Mathematical transformation:
        gamma_hat_j = gamma_j / s_j
        beta_hat_j  = beta_j  / s_j   (if bias exists)

    This allows the normalization layer to directly output smoothed activations
    X_hat during inference, eliminating any runtime per-channel division overhead.

    Parameters:
    -----------
    layernorm : nn.LayerNorm or nn.Module
        Preceding normalization layer with `weight` (gamma) and optional `bias` (beta).
    smoothing_scale : torch.Tensor
        1D smoothing scale vector s of shape (C_in,).
    """
    if not hasattr(layernorm, "weight") or layernorm.weight is None:
        raise ValueError("Normalization module must possess a 'weight' parameter")

    c_in = layernorm.weight.shape[-1]
    if smoothing_scale.shape[-1] != c_in:
        raise ValueError(
            f"smoothing_scale dimension ({smoothing_scale.shape[-1]}) does not match "
            f"LayerNorm weight dimension ({c_in})"
        )

    with torch.no_grad():
        scale = smoothing_scale.to(device=layernorm.weight.device, dtype=layernorm.weight.dtype)
        layernorm.weight.div_(scale)
        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            layernorm.bias.div_(scale)


# ============================================================================
# Quantization Primitives (INT8 Symmetrical Uniform)
# ============================================================================

def quantize_activation_per_token(
    x: torch.Tensor,
    qmax: int = 127,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-token (per-row) INT8 quantization for activations.

    For each token vector x_b in R^(C_in):
        s_b = max_j |x_{b, j}| / qmax
        q_{b, j} = clip( round( x_{b, j} / s_b ), -qmax, qmax )

    Parameters:
    -----------
    x : torch.Tensor
        Activation tensor of shape (..., C_in).
    qmax : int
        Maximum quantization value (default 127 for signed INT8).
    eps : float
        Numerical epsilon to avoid division by zero.

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]:
        - Quantized tensor in torch.int8 with identical shape as x.
        - Per-token scale tensor of shape (..., 1) in matching floating-point dtype.
    """
    # Max along channel dimension (last dimension)
    abs_max = torch.max(torch.abs(x), dim=-1, keepdim=True).values
    scale = torch.clamp(abs_max / float(qmax), min=eps)

    # Quantize and clip
    q = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_activation_per_token(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize per-token quantized activations back to floating-point.

    Parameters:
    -----------
    q : torch.Tensor
        Quantized integer tensor (..., C_in).
    scale : torch.Tensor
        Scale tensor (..., 1).
    dtype : torch.dtype
        Target floating-point dtype.

    Returns:
    --------
    torch.Tensor
        Dequantized floating-point tensor of shape (..., C_in).
    """
    return q.to(dtype) * scale.to(dtype)


def quantize_weight_per_channel(
    w: torch.Tensor,
    qmax: int = 127,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-channel (per-output-row) INT8 quantization for linear weights.

    For weight matrix W of shape (C_out, C_in):
        s_i = max_j |W_{i, j}| / qmax
        Q_{i, j} = clip( round( W_{i, j} / s_i ), -qmax, qmax )

    Parameters:
    -----------
    w : torch.Tensor
        Weight matrix of shape (C_out, C_in).
    qmax : int
        Maximum quantization value (default: 127 for INT8).
    eps : float
        Numerical stability epsilon.

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]:
        - Quantized weight tensor in torch.int8 of shape (C_out, C_in).
        - Scale tensor of shape (C_out, 1) in weight's original dtype.
    """
    if w.ndim != 2:
        raise ValueError(f"Weight must be a 2D tensor of shape (C_out, C_in), got shape {w.shape}")

    abs_max = torch.max(torch.abs(w), dim=1, keepdim=True).values
    scale = torch.clamp(abs_max / float(qmax), min=eps)

    q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_weight_per_channel(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize per-channel quantized weights back to floating-point.

    Parameters:
    -----------
    q : torch.Tensor
        Quantized weight tensor in torch.int8 of shape (C_out, C_in).
    scale : torch.Tensor
        Per-row scale tensor of shape (C_out, 1).
    dtype : torch.dtype
        Target floating-point dtype.

    Returns:
    --------
    torch.Tensor
        Dequantized weight matrix of shape (C_out, C_in).
    """
    return q.to(dtype) * scale.to(dtype)


# ============================================================================
# Outlier Clipping Primitive
# ============================================================================

def clip_activation_outliers(
    x: torch.Tensor,
    clipping_ratio: float = 0.999,
    method: str = "percentile",
) -> torch.Tensor:
    """
    Outlier-aware activation clipping to bound non-systematic activation spikes.

    Parameters:
    -----------
    x : torch.Tensor
        Input activation tensor.
    clipping_ratio : float
        Quantile/percentile threshold in (0.0, 1.0], or multiplier for stddev.
        Default: 0.999 (99.9th percentile).
    method : str
        Clipping strategy: "percentile" or "stddev".

    Returns:
    --------
    torch.Tensor
        Clipped activation tensor with preserved dtype and shape.
    """
    if method == "percentile":
        if not (0.0 < clipping_ratio <= 1.0):
            raise ValueError(f"clipping_ratio must be in (0.0, 1.0], got {clipping_ratio}")
        abs_x = torch.abs(x)
        # Calculate threshold at requested percentile
        k = int(abs_x.numel() * clipping_ratio)
        k = max(1, min(k, abs_x.numel()))
        threshold = torch.kthvalue(abs_x.reshape(-1), k).values
        return torch.clamp(x, -threshold, threshold)
    elif method == "stddev":
        mean = torch.mean(x)
        std = torch.std(x)
        bound = clipping_ratio * std
        return torch.clamp(x, mean - bound, mean + bound)
    else:
        raise ValueError(f"Unknown clipping method: {method}. Choose 'percentile' or 'stddev'.")


# ============================================================================
# Diagnostic Analysis Suite
# ============================================================================

def analyze_activation_outliers(
    x: torch.Tensor,
    threshold: float = 6.0,
    dim: int = -1,
) -> OutlierDiagnostics:
    """
    Analyze activation distribution to measure kurtosis, channel peaks, and identify outliers.

    Parameters:
    -----------
    x : torch.Tensor
        Activation tensor of shape (..., C).
    threshold : float
        Outlier magnitude threshold (default: 6.0).
    dim : int
        Channel feature dimension (default: -1).

    Returns:
    --------
    OutlierDiagnostics
        Structured dataclass containing channel peaks, kurtosis, PAR, and outlier indices.
    """
    orig_dim = dim if dim >= 0 else x.ndim + dim
    c = x.shape[orig_dim]

    # Permute target dimension to last, reshape to (N, C)
    if orig_dim != x.ndim - 1:
        perm = list(range(x.ndim))
        perm.pop(orig_dim)
        perm.append(orig_dim)
        x_flat = x.permute(*perm).reshape(-1, c)
    else:
        x_flat = x.reshape(-1, c)

    abs_x = torch.abs(x_flat)
    channel_peaks = torch.max(abs_x, dim=0).values  # Shape (C,)

    global_max = float(torch.max(abs_x).item())
    global_mean = float(torch.mean(abs_x).item())
    par = global_max / (global_mean + 1e-12)

    # Compute kurtosis of channel peaks
    mean_peak = torch.mean(channel_peaks)
    std_peak = torch.std(channel_peaks) + 1e-12
    m4 = torch.mean((channel_peaks - mean_peak) ** 4)
    kurtosis = float((m4 / (std_peak ** 4)).item())

    # Identify outlier channels
    outlier_mask = channel_peaks >= threshold
    outlier_indices = torch.nonzero(outlier_mask, as_tuple=False).squeeze(-1)
    if outlier_indices.ndim == 0 and outlier_indices.numel() == 1:
        outlier_indices = outlier_indices.unsqueeze(0)

    outlier_ratio = float(outlier_indices.numel() / c)

    return OutlierDiagnostics(
        num_channels=c,
        channel_peaks=channel_peaks,
        global_max=global_max,
        global_mean=global_mean,
        par=par,
        kurtosis=kurtosis,
        outlier_channel_indices=outlier_indices,
        outlier_ratio=outlier_ratio,
    )


# ============================================================================
# Calibration Observer
# ============================================================================

class SmoothQuantObserver:
    """
    Running activation observer that accumulates per-channel maximum absolute
    activations across calibration batches for optimal scale migration computation.
    """

    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        self.channel_max: Optional[torch.Tensor] = None
        self.batch_count = 0

    def update(self, x: torch.Tensor) -> None:
        """
        Update running channel maximums with a new calibration batch.

        Parameters:
        -----------
        x : torch.Tensor
            Activation tensor of shape (..., C_in).
        """
        c_in = x.shape[-1]
        if c_in != self.num_channels:
            raise ValueError(
                f"Input channel count ({c_in}) does not match "
                f"observer initialized channels ({self.num_channels})"
            )

        x_2d = x.reshape(-1, c_in)
        batch_max = torch.max(torch.abs(x_2d), dim=0).values.detach()

        if self.channel_max is None:
            self.channel_max = batch_max.clone()
        else:
            self.channel_max = torch.maximum(self.channel_max, batch_max.to(self.channel_max.device))

        self.batch_count += 1

    def get_act_scales(self) -> torch.Tensor:
        """Get the accumulated per-channel activation maximums."""
        if self.channel_max is None:
            raise RuntimeError("Observer has not processed any calibration batches yet.")
        return self.channel_max

    def compute_scales(
        self,
        weight: torch.Tensor,
        alpha: float = 0.5,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        Compute optimal SmoothQuant migration scales for a given weight tensor.

        Parameters:
        -----------
        weight : torch.Tensor
            Weight tensor of shape (C_out, C_in).
        alpha : float
            Migration strength parameter in [0.0, 1.0].
        eps : float
            Numerical stability constant.

        Returns:
        --------
        torch.Tensor
            Smoothing scale vector s of shape (C_in,).
        """
        act_scales = self.get_act_scales()
        return compute_smoothquant_scales(act_scales, weight, alpha=alpha, eps=eps)

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.channel_max = None
        self.batch_count = 0


# ============================================================================
# Quantized GEMM Execution Engine (W8A8 Matrix Multiplication)
# ============================================================================

def smoothquant_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    smoothing_scale: Optional[torch.Tensor] = None,
    act_quant_per_token: bool = True,
    weight_quant_per_channel: bool = True,
    qmax: int = 127,
) -> torch.Tensor:
    """
    Execute end-to-end W8A8 quantized matrix multiplication with SmoothQuant.

    Mathematical flow:
    1. Smooth activations:
           X_hat = X / s   (or X if s is already folded into previous layer)
    2. Smooth weights:
           W_hat = W * s   (or pre-smoothed W)
    3. Quantize X_hat to INT8 (per-token or per-tensor):
           Q_X, s_X = quantize(X_hat)
    4. Quantize W_hat to INT8 (per-channel or per-tensor):
           Q_W, s_W = quantize(W_hat)
    5. Execute INT8 Matrix Multiplication:
           Z = Q_X @ Q_W^T   (in int32 accumulator)
    6. Dequantize to float:
           Y = Z * (s_X * s_W^T) + bias

    Parameters:
    -----------
    x : torch.Tensor
        Activation tensor of shape (..., C_in).
    weight : torch.Tensor
        Weight matrix of shape (C_out, C_in).
    bias : Optional[torch.Tensor]
        Optional bias vector of shape (C_out,).
    smoothing_scale : Optional[torch.Tensor]
        Optional 1D smoothing scale vector s of shape (C_in,).
    act_quant_per_token : bool
        Whether to use per-token activation scaling (default: True).
    weight_quant_per_channel : bool
        Whether to use per-channel weight scaling (default: True).
    qmax : int
        Maximum quantization value (default: 127).

    Returns:
    --------
    torch.Tensor
        Computed output tensor of shape (..., C_out) with original input dtype.
    """
    orig_shape = x.shape
    c_in = orig_shape[-1]
    c_out = weight.shape[0]
    out_shape = orig_shape[:-1] + (c_out,)
    input_dtype = x.dtype

    # Flatten leading batch/sequence dimensions to 2D
    x_2d = x.reshape(-1, c_in)

    # 1. Apply smoothing if scale provided
    if smoothing_scale is not None:
        x_smooth = apply_activation_smoothing(x_2d, smoothing_scale)
        w_smooth = apply_weight_smoothing(weight, smoothing_scale)
    else:
        x_smooth = x_2d
        w_smooth = weight

    # 2. Quantize activations to INT8
    if act_quant_per_token:
        q_x, s_x = quantize_activation_per_token(x_smooth, qmax=qmax)
    else:
        abs_max_x = torch.max(torch.abs(x_smooth))
        s_x = torch.clamp(abs_max_x / float(qmax), min=1e-8)
        q_x = torch.clamp(torch.round(x_smooth / s_x), -qmax, qmax).to(torch.int8)

    # 3. Quantize weights to INT8
    if weight_quant_per_channel:
        q_w, s_w = quantize_weight_per_channel(w_smooth, qmax=qmax)
    else:
        abs_max_w = torch.max(torch.abs(w_smooth))
        s_w = torch.clamp(abs_max_w / float(qmax), min=1e-8)
        q_w = torch.clamp(torch.round(w_smooth / s_w), -qmax, qmax).to(torch.int8)

    # 4. Integer GEMM: (M, C_in) x (C_out, C_in)^T -> (M, C_out)
    # Using int32 intermediate accumulation
    z_int = torch.matmul(q_x.to(torch.float32), q_w.to(torch.float32).t())

    # 5. Dequantize
    # s_x has shape (M, 1) or scalar; s_w has shape (C_out, 1) or scalar
    if act_quant_per_token and weight_quant_per_channel:
        # Scale matrix is s_x @ s_w^T: (M, 1) * (1, C_out) -> (M, C_out)
        combined_scale = s_x.to(torch.float32) * s_w.to(torch.float32).t()
        y_2d = z_int * combined_scale
    elif act_quant_per_token and not weight_quant_per_channel:
        y_2d = z_int * (s_x.to(torch.float32) * float(s_w))
    elif not act_quant_per_token and weight_quant_per_channel:
        y_2d = z_int * (float(s_x) * s_w.to(torch.float32).t())
    else:
        y_2d = z_int * float(s_x * s_w)

    # 6. Add bias if present
    if bias is not None:
        y_2d = y_2d + bias.to(y_2d.device, dtype=torch.float32)

    return y_2d.reshape(out_shape).to(input_dtype)


# ============================================================================
# Autograd Function with Straight-Through Estimator (STE)
# ============================================================================

class _SmoothQuantLinearFunction(torch.autograd.Function):
    """
    Autograd function implementing SmoothQuant W8A8 forward pass with
    Straight-Through Estimator (STE) backward gradient propagation for QAT.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        smoothing_scale: Optional[torch.Tensor],
        act_quant_per_token: bool,
        weight_quant_per_channel: bool,
        qmax: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, bias, smoothing_scale)
        ctx.act_quant_per_token = act_quant_per_token
        ctx.weight_quant_per_channel = weight_quant_per_channel
        ctx.qmax = qmax

        return smoothquant_gemm(
            x=x,
            weight=weight,
            bias=bias,
            smoothing_scale=smoothing_scale,
            act_quant_per_token=act_quant_per_token,
            weight_quant_per_channel=weight_quant_per_channel,
            qmax=qmax,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias, smoothing_scale = ctx.saved_tensors
        grad_x = None
        grad_weight = None
        grad_bias = None

        orig_shape = x.shape
        c_in = orig_shape[-1]
        grad_output_2d = grad_output.reshape(-1, weight.shape[0])
        x_2d = x.reshape(-1, c_in)

        # Straight-Through Estimator: propagate gradients through quantized layer
        if ctx.needs_input_grad[0]:
            grad_x_2d = torch.matmul(grad_output_2d, weight)
            grad_x = grad_x_2d.reshape(orig_shape)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output_2d.t(), x_2d)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output_2d, dim=0)

        # smoothing_scale, flags, qmax do not require gradients
        return grad_x, grad_weight, grad_bias, None, None, None, None


def smoothquant_linear_ste(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    smoothing_scale: Optional[torch.Tensor] = None,
    act_quant_per_token: bool = True,
    weight_quant_per_channel: bool = True,
    qmax: int = 127,
) -> torch.Tensor:
    """Wrapper calling `_SmoothQuantLinearFunction.apply` with STE gradient propagation."""
    return _SmoothQuantLinearFunction.apply(
        x,
        weight,
        bias,
        smoothing_scale,
        act_quant_per_token,
        weight_quant_per_channel,
        qmax,
    )


# ============================================================================
# Object-Oriented Quantizer (SmoothQuantQuantizer)
# ============================================================================

class SmoothQuantQuantizer(BaseQuantizer):
    """
    Object-oriented quantizer for Outlier-Aware Activation Quantization (SmoothQuant).

    Inherits from `BaseQuantizer`. Implements channel scale migration, activation
    smoothing, quantization, dequantization, and outlier diagnostics.

    Parameters:
    -----------
    bits : int
        Quantization bit-width (default: 8).
    alpha : float
        Migration strength parameter in [0.0, 1.0] (default: 0.5).
    per_token : bool
        Whether to apply per-token activation quantization (default: True).
    eps : float
        Numerical epsilon for scale clamping (default: 1e-5).
    """

    def __init__(
        self,
        bits: int = 8,
        alpha: float = 0.5,
        per_token: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__(bits=bits)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        self.alpha = alpha
        self.per_token = per_token
        self.eps = eps
        self.qmax = (1 << (bits - 1)) - 1  # For 8 bits: 127

    def quantize(
        self,
        x: torch.Tensor,
        smoothing_scale: Optional[torch.Tensor] = None,
    ) -> SmoothQuantState:
        """
        Smooth and quantize activation tensor into `SmoothQuantState`.

        Parameters:
        -----------
        x : torch.Tensor
            Activation tensor of shape (..., C_in).
        smoothing_scale : Optional[torch.Tensor]
            1D smoothing scale vector s of shape (C_in,).
            If None, scales are derived dynamically from batch statistics.

        Returns:
        --------
        SmoothQuantState
            Dataclass holding smoothed activations, quantized integers, scales, and metadata.
        """
        orig_shape = x.shape
        c_in = orig_shape[-1]
        x_flat = x.reshape(-1, c_in)

        # Dynamic scale derivation if not explicitly supplied
        if smoothing_scale is None:
            act_max = torch.max(torch.abs(x_flat), dim=0).values
            # Dynamic scale: s_j = act_max^alpha
            smoothing_scale = torch.clamp(
                torch.pow(torch.clamp(act_max, min=self.eps), self.alpha),
                min=self.eps,
            )

        # Apply smoothing transform
        smoothed_x = apply_activation_smoothing(x_flat, smoothing_scale)

        # Quantize smoothed activations
        if self.per_token:
            q_x, act_scale = quantize_activation_per_token(smoothed_x, qmax=self.qmax)
        else:
            abs_max = torch.max(torch.abs(smoothed_x))
            act_scale = torch.clamp(abs_max / float(self.qmax), min=1e-8)
            q_x = torch.clamp(torch.round(smoothed_x / act_scale), -self.qmax, self.qmax).to(torch.int8)

        return SmoothQuantState(
            smoothed_x=smoothed_x.reshape(orig_shape),
            quantized_x=q_x.reshape(orig_shape),
            act_scale=act_scale,
            migration_scale=smoothing_scale,
            original_shape=orig_shape,
            alpha=self.alpha,
            input_dtype=x.dtype,
        )

    def dequantize(self, state: SmoothQuantState) -> torch.Tensor:
        """
        Reconstruct full floating-point activation tensor from `SmoothQuantState`.

        Parameters:
        -----------
        state : SmoothQuantState
            Quantized state container.

        Returns:
        --------
        torch.Tensor
            Reconstructed activation tensor with original shape and input dtype.
        """
        c_in = state.original_shape[-1]
        q_2d = state.quantized_x.reshape(-1, c_in)

        # 1. Dequantize to smoothed float: X_hat = Q_X * s_X
        if state.act_scale.ndim >= 1 and state.act_scale.numel() > 1:
            dequant_smoothed = dequantize_activation_per_token(
                q_2d, state.act_scale, dtype=state.input_dtype
            )
        else:
            dequant_smoothed = q_2d.to(state.input_dtype) * state.act_scale.to(state.input_dtype)

        # 2. Invert smoothing transform: X = X_hat * s
        scale = state.migration_scale.to(device=dequant_smoothed.device, dtype=state.input_dtype)
        x_recon = dequant_smoothed * scale.unsqueeze(0)

        return x_recon.reshape(state.original_shape)


# ============================================================================
# Drop-in PyTorch Neural Network Module (SmoothQuantLinear)
# ============================================================================

class SmoothQuantLinear(nn.Module):
    """
    Drop-in replacement for `nn.Linear` implementing SmoothQuant W8A8 quantization.

    Supports:
    - Pre-computed static smoothing scales (from calibration) or dynamic smoothing
    - Offline folding into preceding LayerNorm (`fold_into_layernorm`)
    - Both fake-quantized and integer GEMM forward execution
    - PyTorch autograd Straight-Through Estimator (STE) for fine-tuning
    - Conversion factory `from_linear(linear, act_scales, alpha=0.5)`
    - Parameter memory profiling (`memory_stats`)

    Parameters:
    -----------
    in_features : int
        Number of input channels (C_in).
    out_features : int
        Number of output channels (C_out).
    bias : bool
        Whether to include an additive bias parameter (default: True).
    alpha : float
        Migration strength parameter in [0.0, 1.0] (default: 0.5).
    act_quant_per_token : bool
        Whether to quantize activations per-token (default: True).
    weight_quant_per_channel : bool
        Whether to quantize weights per-channel (default: True).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 0.5,
        act_quant_per_token: bool = True,
        weight_quant_per_channel: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.act_quant_per_token = act_quant_per_token
        self.weight_quant_per_channel = weight_quant_per_channel

        # Parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Smoothing scale buffer s in R^(C_in)
        self.register_buffer("smoothing_scale", None)

        # Pre-quantized integer weights & scales (for static inference)
        self.register_buffer("int_weight", None)
        self.register_buffer("weight_scale", None)

        # Flag indicating whether smoothing was folded into preceding LayerNorm
        self.is_folded: bool = False

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        act_scales: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        quantize_weights: bool = True,
    ) -> "SmoothQuantLinear":
        """
        Factory method constructing a `SmoothQuantLinear` layer from a standard `nn.Linear`.

        Parameters:
        -----------
        linear : nn.Linear
            Source PyTorch linear layer.
        act_scales : Optional[torch.Tensor]
            Pre-computed per-channel activation maximums from calibration data.
        alpha : float
            Migration strength parameter in [0.0, 1.0] (default: 0.5).
        quantize_weights : bool
            Whether to immediately pre-quantize weights into INT8 buffers.

        Returns:
        --------
        SmoothQuantLinear
            Drop-in replacement module initialized with weights from `linear`.
        """
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        module = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            alpha=alpha,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if has_bias:
                module.bias.copy_(linear.bias)

            if act_scales is not None:
                # Compute and store smoothing scale vector s
                s = compute_smoothquant_scales(act_scales, module.weight, alpha=alpha)
                module.smoothing_scale = s.to(device=linear.weight.device)

                if quantize_weights:
                    module.prepare_quantized_weights()

        return module

    def prepare_quantized_weights(self) -> None:
        """Smooth and pre-quantize weights into static INT8 buffers."""
        with torch.no_grad():
            if self.smoothing_scale is not None and not self.is_folded:
                w_smooth = apply_weight_smoothing(self.weight, self.smoothing_scale)
            else:
                w_smooth = self.weight

            if self.weight_quant_per_channel:
                q_w, s_w = quantize_weight_per_channel(w_smooth)
            else:
                abs_max = torch.max(torch.abs(w_smooth))
                s_w = torch.clamp(abs_max / 127.0, min=1e-8)
                q_w = torch.clamp(torch.round(w_smooth / s_w), -127, 127).to(torch.int8)

            self.int_weight = q_w
            self.weight_scale = s_w

    def fold_into_layernorm(self, layernorm: Union[nn.LayerNorm, nn.Module]) -> None:
        """
        Fold smoothing scale s into preceding LayerNorm, enabling zero-overhead inference.

        Parameters:
        -----------
        layernorm : nn.LayerNorm or nn.Module
            Preceding normalization module.
        """
        if self.smoothing_scale is None:
            raise RuntimeError("Cannot fold into LayerNorm: smoothing_scale has not been set.")

        fold_smoothing_into_layernorm(layernorm, self.smoothing_scale)
        # Weight is smoothed and stored permanently
        with torch.no_grad():
            self.weight.copy_(apply_weight_smoothing(self.weight, self.smoothing_scale))
        self.is_folded = True
        self.prepare_quantized_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass.

        If `is_folded` is True, activation smoothing was already absorbed into
        the preceding LayerNorm, so x is fed directly to INT8 GEMM.
        Otherwise, smoothing is applied using `smoothing_scale`.
        """
        scale_to_apply = None if self.is_folded else self.smoothing_scale

        if self.training:
            return smoothquant_linear_ste(
                x=x,
                weight=self.weight,
                bias=self.bias,
                smoothing_scale=scale_to_apply,
                act_quant_per_token=self.act_quant_per_token,
                weight_quant_per_channel=self.weight_quant_per_channel,
            )
        else:
            return smoothquant_gemm(
                x=x,
                weight=self.weight,
                bias=self.bias,
                smoothing_scale=scale_to_apply,
                act_quant_per_token=self.act_quant_per_token,
                weight_quant_per_channel=self.weight_quant_per_channel,
            )

    def memory_stats(self) -> Dict[str, Union[int, float]]:
        """
        Calculate parameter memory footprint statistics.

        Returns:
        --------
        Dict with full precision bytes, INT8 quantized bytes, and compression ratio.
        """
        num_weights = self.weight.numel()
        bias_bytes = self.bias.numel() * 4 if self.bias is not None else 0

        fp32_bytes = num_weights * 4 + bias_bytes
        fp16_bytes = num_weights * 2 + bias_bytes

        # INT8 weights (1 byte/param) + scale vector (C_out * 4 bytes) + bias (FP32)
        int8_weight_bytes = num_weights * 1
        scale_bytes = (self.out_features * 4) if self.weight_quant_per_channel else 4
        smoothing_bytes = (self.in_features * 4) if (self.smoothing_scale is not None and not self.is_folded) else 0

        int8_total_bytes = int8_weight_bytes + scale_bytes + smoothing_bytes + bias_bytes

        return {
            "num_parameters": num_weights + (self.bias.numel() if self.bias is not None else 0),
            "fp32_bytes": fp32_bytes,
            "fp16_bytes": fp16_bytes,
            "int8_total_bytes": int8_total_bytes,
            "compression_ratio_vs_fp32": fp32_bytes / max(1, int8_total_bytes),
            "compression_ratio_vs_fp16": fp16_bytes / max(1, int8_total_bytes),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, alpha={self.alpha}, is_folded={self.is_folded}"
        )


# ============================================================================
# Benchmarking & Comparison Suite
# ============================================================================

def compare_smoothquant_vs_standard_8bit(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark and compare output quality of Full Precision (FP32), Standard INT8
    (per-tensor & per-token), and SmoothQuant INT8 on activation tensors with outliers.

    Parameters:
    -----------
    x : torch.Tensor
        Input activations with emergent kurtic outliers.
    weight : torch.Tensor
        Linear weight matrix.
    bias : Optional[torch.Tensor]
        Optional linear bias.
    alpha : float
        SmoothQuant migration parameter (default: 0.5).

    Returns:
    --------
    Dict containing output SNR (dB), Cosine Similarity, MAE, and Peak Error
    for each quantization scheme.
    """
    # 1. Full-precision reference output Y_fp
    y_fp = F.linear(x.to(torch.float32), weight.to(torch.float32), bias.to(torch.float32) if bias is not None else None)

    # 2. Standard INT8 (Per-Token Activation, Per-Channel Weight) WITHOUT smoothing
    y_std_int8 = smoothquant_gemm(
        x=x,
        weight=weight,
        bias=bias,
        smoothing_scale=None,
        act_quant_per_token=True,
        weight_quant_per_channel=True,
    )

    # 3. Standard INT8 (Per-Tensor Activation, Per-Tensor Weight) WITHOUT smoothing
    y_std_pertensor = smoothquant_gemm(
        x=x,
        weight=weight,
        bias=bias,
        smoothing_scale=None,
        act_quant_per_token=False,
        weight_quant_per_channel=False,
    )

    # 4. SmoothQuant INT8 (Dynamic smoothing with alpha)
    act_scales = torch.max(torch.abs(x.reshape(-1, x.shape[-1])), dim=0).values
    s = compute_smoothquant_scales(act_scales, weight, alpha=alpha)

    y_smoothquant = smoothquant_gemm(
        x=x,
        weight=weight,
        bias=bias,
        smoothing_scale=s,
        act_quant_per_token=True,
        weight_quant_per_channel=True,
    )

    # Evaluation metric calculator
    def _metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        diff = y_pred.to(torch.float32) - y_true.to(torch.float32)
        noise_var = torch.mean(diff ** 2).item()
        signal_var = torch.mean(y_true.to(torch.float32) ** 2).item()
        snr_db = 10.0 * math.log10(max(signal_var, 1e-12) / max(noise_var, 1e-12))

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            y_pred.reshape(-1).unsqueeze(0).to(torch.float32),
            y_true.reshape(-1).unsqueeze(0).to(torch.float32),
        ).item()

        mae = torch.mean(torch.abs(diff)).item()
        max_err = torch.max(torch.abs(diff)).item()

        return {
            "snr_db": round(snr_db, 2),
            "cosine_similarity": round(cos_sim, 6),
            "mae": round(mae, 6),
            "max_error": round(max_err, 6),
        }

    return {
        "standard_int8_per_tensor": _metrics(y_std_pertensor, y_fp),
        "standard_int8_per_token": _metrics(y_std_int8, y_fp),
        "smoothquant_w8a8": _metrics(y_smoothquant, y_fp),
    }
