"""
Comprehensive unit tests for Outlier-Aware Activation Quantization (SmoothQuant).

Category 7: LLM-Specific Quantization (Milestone 7).
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ============================================================================
# Helper Functions
# ============================================================================

def make_synthetic_outlier_activation(
    batch_size: int = 16,
    seq_len: int = 32,
    hidden_dim: int = 128,
    outlier_channels: Tuple[int, ...] = (3, 17, 85),
    outlier_magnitude: float = 60.0,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate synthetic transformer activation tensor with kurtic channel outliers."""
    torch.manual_seed(seed)
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device) * 0.4
    for c in outlier_channels:
        x[:, :, c] += outlier_magnitude * torch.sign(torch.randn(batch_size, seq_len, device=device))
    return x


# ============================================================================
# Unit Tests
# ============================================================================

def test_compute_smoothquant_scales_math():
    """Verify mathematical correctness of SmoothQuant scale calculation: s_j = act^alpha / weight^(1-alpha)."""
    act_scales = torch.tensor([10.0, 4.0, 1.0, 0.25], dtype=torch.float32)
    weight = torch.tensor([
        [1.0, 2.0, 0.5, 0.1],
        [2.0, 1.0, 0.5, 0.2],
    ], dtype=torch.float32)

    # Max weight per channel: [2.0, 2.0, 0.5, 0.2]
    # For alpha = 0.5: s_j = sqrt(act) / sqrt(weight)
    s_half = compute_smoothquant_scales(act_scales, weight, alpha=0.5)
    expected_half = torch.sqrt(act_scales) / torch.sqrt(torch.tensor([2.0, 2.0, 0.5, 0.2]))
    assert torch.allclose(s_half, expected_half, atol=1e-5)

    # For alpha = 1.0: s_j = act / 1.0 = act
    s_one = compute_smoothquant_scales(act_scales, weight, alpha=1.0)
    assert torch.allclose(s_one, act_scales, atol=1e-5)

    # For alpha = 0.0: s_j = 1.0 / weight
    s_zero = compute_smoothquant_scales(act_scales, weight, alpha=0.0)
    expected_zero = 1.0 / torch.tensor([2.0, 2.0, 0.5, 0.2])
    assert torch.allclose(s_zero, expected_zero, atol=1e-5)


def test_compute_smoothquant_scales_validation():
    """Verify input validation and edge cases."""
    act_scales = torch.tensor([1.0, 2.0, 3.0])
    weight = torch.randn(4, 3)

    # Invalid alpha
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_smoothquant_scales(act_scales, weight, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_smoothquant_scales(act_scales, weight, alpha=1.5)

    # Dimension mismatch
    with pytest.raises(ValueError, match="must match"):
        compute_smoothquant_scales(act_scales, torch.randn(4, 5))


def test_smoothing_mathematical_equivalence():
    """Verify mathematical identity: X @ W^T == (X / s) @ (W * s)^T."""
    torch.manual_seed(123)
    x = torch.randn(8, 64)
    w = torch.randn(32, 64)
    s = torch.rand(64) + 0.5  # Positive scaling vector

    # Original product
    y_orig = torch.matmul(x, w.t())

    # Smoothed product
    x_smooth = apply_activation_smoothing(x, s)
    w_smooth = apply_weight_smoothing(w, s)
    y_smooth = torch.matmul(x_smooth, w_smooth.t())

    assert torch.allclose(y_orig, y_smooth, atol=1e-5, rtol=1e-5)


def test_fold_smoothing_into_layernorm():
    """Verify that folding scale into LayerNorm produces exactly smoothed activations."""
    torch.manual_seed(42)
    dim = 64
    ln = nn.LayerNorm(dim)
    x = torch.randn(4, 16, dim)
    s = torch.rand(dim) + 0.2

    # Compute expected smoothed output using explicit division
    y_expected = ln(x) / s

    # Fold into LayerNorm
    fold_smoothing_into_layernorm(ln, s)
    y_folded = ln(x)

    assert torch.allclose(y_folded, y_expected, atol=1e-5, rtol=1e-5)


def test_quantize_dequantize_activation_per_token():
    """Verify per-token activation quantization round-trip fidelity."""
    torch.manual_seed(77)
    x = torch.randn(10, 128)

    q, scale = quantize_activation_per_token(x, qmax=127)
    assert q.dtype == torch.int8
    assert q.shape == x.shape
    assert scale.shape == (10, 1)

    x_rec = dequantize_activation_per_token(q, scale)
    # Cosine fidelity should be > 0.999
    cos = F.cosine_similarity(x.reshape(-1), x_rec.reshape(-1), dim=0)
    assert cos.item() > 0.999


def test_quantize_dequantize_weight_per_channel():
    """Verify per-channel weight quantization round-trip fidelity."""
    torch.manual_seed(88)
    w = torch.randn(64, 128)

    q, scale = quantize_weight_per_channel(w, qmax=127)
    assert q.dtype == torch.int8
    assert q.shape == w.shape
    assert scale.shape == (64, 1)

    w_rec = dequantize_weight_per_channel(q, scale)
    cos = F.cosine_similarity(w.reshape(-1), w_rec.reshape(-1), dim=0)
    assert cos.item() > 0.999


def test_clip_activation_outliers():
    """Verify percentile and stddev outlier clipping."""
    torch.manual_seed(99)
    x = torch.randn(100, 100)
    # Inject an extreme anomaly
    x[0, 0] = 500.0
    x[1, 1] = -500.0

    # Percentile clipping
    x_clipped_perc = clip_activation_outliers(x, clipping_ratio=0.99, method="percentile")
    assert torch.max(torch.abs(x_clipped_perc)) < 50.0

    # Stddev clipping
    x_clipped_std = clip_activation_outliers(x, clipping_ratio=3.0, method="stddev")
    assert torch.max(torch.abs(x_clipped_std)) < 50.0


def test_analyze_activation_outliers():
    """Verify diagnostic detection of kurtosis and outlier channels."""
    outlier_channels = (5, 42, 99)
    x = make_synthetic_outlier_activation(
        batch_size=8,
        seq_len=16,
        hidden_dim=128,
        outlier_channels=outlier_channels,
        outlier_magnitude=80.0,
    )

    diag = analyze_activation_outliers(x, threshold=10.0, dim=-1)
    assert diag.num_channels == 128
    assert diag.global_max > 70.0
    assert diag.par > 10.0
    assert diag.kurtosis > 5.0
    assert len(diag.outlier_channel_indices) == len(outlier_channels)
    for c in outlier_channels:
        assert c in diag.outlier_channel_indices.tolist()


def test_smoothquant_observer():
    """Verify multi-batch activation calibration observer."""
    torch.manual_seed(101)
    observer = SmoothQuantObserver(num_channels=64)

    # Process 3 minibatches
    batch1 = torch.randn(4, 10, 64) * 2.0
    batch2 = torch.randn(2, 5, 64) * 3.0
    batch3 = torch.randn(8, 20, 64) * 1.5

    observer.update(batch1)
    observer.update(batch2)
    observer.update(batch3)

    assert observer.batch_count == 3
    act_scales = observer.get_act_scales()
    assert act_scales.shape == (64,)

    # Verify that act_scales is >= any batch's per-channel max
    m1 = torch.max(torch.abs(batch1.reshape(-1, 64)), dim=0).values
    m2 = torch.max(torch.abs(batch2.reshape(-1, 64)), dim=0).values
    m3 = torch.max(torch.abs(batch3.reshape(-1, 64)), dim=0).values
    expected_max = torch.maximum(torch.maximum(m1, m2), m3)
    assert torch.allclose(act_scales, expected_max, atol=1e-5)

    # Compute scales with weight
    w = torch.randn(32, 64)
    s = observer.compute_scales(w, alpha=0.5)
    assert s.shape == (64,)
    assert torch.all(s > 0)


def test_smoothquant_gemm_shapes_and_bias():
    """Test smoothquant_gemm with 2D, 3D, and 4D inputs, with and without bias."""
    torch.manual_seed(202)
    w = torch.randn(32, 48)
    b = torch.randn(32)
    s = torch.rand(48) + 0.5

    # 2D input (B, C_in)
    x_2d = torch.randn(8, 48)
    y_2d = smoothquant_gemm(x_2d, w, bias=b, smoothing_scale=s)
    assert y_2d.shape == (8, 32)

    # 3D input (B, S, C_in)
    x_3d = torch.randn(4, 16, 48)
    y_3d = smoothquant_gemm(x_3d, w, bias=b, smoothing_scale=s)
    assert y_3d.shape == (4, 16, 32)

    # 4D input (B, H, S, C_in) without bias
    x_4d = torch.randn(2, 4, 8, 48)
    y_4d = smoothquant_gemm(x_4d, w, bias=None, smoothing_scale=s)
    assert y_4d.shape == (2, 4, 8, 32)


def test_smoothquant_quantizer_oop():
    """Test SmoothQuantQuantizer object-oriented quantizer and dequantizer."""
    torch.manual_seed(303)
    quantizer = SmoothQuantQuantizer(bits=8, alpha=0.5, per_token=True)

    x = make_synthetic_outlier_activation(
        batch_size=4,
        seq_len=8,
        hidden_dim=32,
        outlier_channels=(2, 10),
        outlier_magnitude=40.0,
    )

    state = quantizer.quantize(x)
    assert isinstance(state, SmoothQuantState)
    assert state.quantized_x.dtype == torch.int8
    assert state.quantized_x.shape == x.shape
    assert state.migration_scale.shape == (32,)

    x_rec = quantizer.dequantize(state)
    assert x_rec.shape == x.shape
    # Cosine fidelity should be high (> 0.999)
    cos = F.cosine_similarity(x.reshape(-1), x_rec.reshape(-1), dim=0)
    assert cos.item() > 0.999


def test_smoothquant_linear_module_lifecycle():
    """Test SmoothQuantLinear initialization, factory from_linear, folding, and forward."""
    torch.manual_seed(404)
    linear = nn.Linear(64, 32)
    act_scales = torch.rand(64) * 10.0 + 1.0

    # Create module via factory
    sq_linear = SmoothQuantLinear.from_linear(linear, act_scales=act_scales, alpha=0.5)
    assert sq_linear.smoothing_scale is not None
    assert sq_linear.smoothing_scale.shape == (64,)
    assert sq_linear.int_weight is not None
    assert sq_linear.int_weight.dtype == torch.int8

    # Forward pass in eval mode
    sq_linear.eval()
    x = torch.randn(4, 16, 64)
    y = sq_linear(x)
    assert y.shape == (4, 16, 32)

    # Test folding into LayerNorm
    ln = nn.LayerNorm(64)
    sq_linear.fold_into_layernorm(ln)
    assert sq_linear.is_folded is True

    # After folding, forward pass on LN output
    x_ln = ln(x)
    y_folded = sq_linear(x_ln)
    assert y_folded.shape == (4, 16, 32)

    # Memory stats
    mem = sq_linear.memory_stats()
    assert mem["compression_ratio_vs_fp32"] > 3.0


def test_smoothquant_linear_ste_backward():
    """Verify Straight-Through Estimator backward pass gradients."""
    torch.manual_seed(505)
    in_features = 32
    out_features = 16
    x = torch.randn(4, in_features, requires_grad=True)
    w = torch.randn(out_features, in_features, requires_grad=True)
    b = torch.randn(out_features, requires_grad=True)
    s = torch.rand(in_features) + 0.5

    y = smoothquant_linear_ste(x, w, bias=b, smoothing_scale=s)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert w.grad is not None
    assert w.grad.shape == w.shape
    assert b.grad is not None
    assert b.grad.shape == b.shape


def test_compare_smoothquant_vs_standard_8bit_improvement():
    """
    Verify that SmoothQuant provides substantial SNR and fidelity gains
    over standard INT8 on activation tensors with prominent outliers.
    """
    x = make_synthetic_outlier_activation(
        batch_size=16,
        seq_len=32,
        hidden_dim=128,
        outlier_channels=(7, 23, 71),
        outlier_magnitude=60.0,
    )
    w = torch.randn(64, 128) * 0.1
    b = torch.zeros(64)

    results = compare_smoothquant_vs_standard_8bit(x, w, bias=b, alpha=0.5)

    snr_std_tensor = results["standard_int8_per_tensor"]["snr_db"]
    snr_std_token = results["standard_int8_per_token"]["snr_db"]
    snr_smoothquant = results["smoothquant_w8a8"]["snr_db"]

    cos_std_token = results["standard_int8_per_token"]["cosine_similarity"]
    cos_smoothquant = results["smoothquant_w8a8"]["cosine_similarity"]

    # SmoothQuant must achieve superior SNR (> 8 dB gain over standard per-tensor INT8)
    assert snr_smoothquant > snr_std_tensor + 8.0
    assert snr_smoothquant > snr_std_token
    assert cos_smoothquant >= cos_std_token
    assert results["smoothquant_w8a8"]["mae"] < results["standard_int8_per_token"]["mae"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_hardware_execution():
    """Verify SmoothQuant execution on NVIDIA CUDA hardware across FP32, FP16, and BF16."""
    device = "cuda:0"
    w = torch.randn(64, 128, device=device)
    b = torch.randn(64, device=device)
    x = make_synthetic_outlier_activation(
        batch_size=4,
        seq_len=16,
        hidden_dim=128,
        outlier_channels=(3, 15),
        outlier_magnitude=50.0,
        device=device,
    )

    act_scales = torch.max(torch.abs(x.reshape(-1, 128)), dim=0).values
    s = compute_smoothquant_scales(act_scales, w, alpha=0.5)

    # 1. FP32 on CUDA
    y_cuda_fp32 = smoothquant_gemm(x, w, bias=b, smoothing_scale=s)
    assert y_cuda_fp32.device.type == "cuda"
    assert y_cuda_fp32.shape == (4, 16, 64)

    # 2. FP16 on CUDA
    x_fp16 = x.to(torch.float16)
    w_fp16 = w.to(torch.float16)
    b_fp16 = b.to(torch.float16)
    s_fp16 = s.to(torch.float16)
    y_cuda_fp16 = smoothquant_gemm(x_fp16, w_fp16, bias=b_fp16, smoothing_scale=s_fp16)
    assert y_cuda_fp16.dtype == torch.float16
    assert y_cuda_fp16.device.type == "cuda"

    # 3. BF16 on CUDA
    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = w.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    s_bf16 = s.to(torch.bfloat16)
    y_cuda_bf16 = smoothquant_gemm(x_bf16, w_bf16, bias=b_bf16, smoothing_scale=s_bf16)
    assert y_cuda_bf16.dtype == torch.bfloat16
    assert y_cuda_bf16.device.type == "cuda"
