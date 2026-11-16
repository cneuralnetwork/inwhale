"""
Unit tests for Mixed Precision Activations (8-bit Base + 16-bit Outliers).

Category 7: LLM-Specific Quantization (Milestone 7).
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def test_find_outlier_channels_threshold():
    """Verify channel identification when activation magnitude exceeds threshold."""
    torch.manual_seed(42)
    # 4 tokens, 8 channels
    x = torch.randn(4, 8) * 0.5
    # Inject outliers in channels 2 and 5
    x[0, 2] = 12.0
    x[3, 5] = -15.0

    outliers = find_outlier_channels(x, threshold=6.0)
    assert outliers.numel() == 2
    assert set(outliers.tolist()) == {2, 5}

    # Test when no element exceeds threshold
    x_small = torch.randn(4, 8) * 0.1
    outliers_none = find_outlier_channels(x_small, threshold=6.0)
    assert outliers_none.numel() == 0

    # Negative threshold should raise ValueError
    with pytest.raises(ValueError):
        find_outlier_channels(x, threshold=-1.0)


def test_detect_outliers_by_quantile():
    """Verify quantile-based dynamic outlier detection."""
    torch.manual_seed(42)
    x = torch.randn(10, 64) * 0.2
    # Inject extreme outliers in 2 specific channels
    x[:, 7] = 25.0
    x[:, 19] = -30.0

    outliers, thresh = detect_outliers_by_quantile(x, quantile=0.95)
    assert thresh > 0.0
    assert 7 in outliers.tolist()
    assert 19 in outliers.tolist()

    with pytest.raises(ValueError):
        detect_outliers_by_quantile(x, quantile=1.5)


def test_split_activation_channels():
    """Verify clean decomposition into outlier and regular tensors."""
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    outlier_idx = torch.tensor([1, 4], dtype=torch.long)

    x_out, x_reg, reg_idx = split_activation_channels(x, outlier_idx, dim=-1)

    assert x_out.shape == (4, 2)
    assert x_reg.shape == (4, 4)
    assert set(reg_idx.tolist()) == {0, 2, 3, 5}

    # Verify values
    torch.testing.assert_close(x_out, x[:, [1, 4]])
    torch.testing.assert_close(x_reg, x[:, [0, 2, 3, 5]])

    # Edge case: No outliers
    empty_idx = torch.empty(0, dtype=torch.long)
    x_out_empty, x_reg_all, reg_all = split_activation_channels(x, empty_idx, dim=-1)
    assert x_out_empty.shape == (4, 0)
    torch.testing.assert_close(x_reg_all, x)
    assert reg_all.numel() == 6

    # Edge case: All channels outliers
    all_idx = torch.arange(6, dtype=torch.long)
    x_out_all, x_reg_empty, reg_empty = split_activation_channels(x, all_idx, dim=-1)
    assert x_reg_empty.shape == (4, 0)
    torch.testing.assert_close(x_out_all, x)
    assert reg_empty.numel() == 0


def test_split_weight_channels():
    """Verify weight matrix decomposition along input feature dimension."""
    w = torch.randn(8, 10)  # out_features=8, in_features=10
    outlier_idx = torch.tensor([2, 7], dtype=torch.long)

    w_out, w_reg = split_weight_channels(w, outlier_idx, in_feature_dim=1)
    assert w_out.shape == (8, 2)
    assert w_reg.shape == (8, 8)
    torch.testing.assert_close(w_out, w[:, [2, 7]])
    torch.testing.assert_close(w_reg, w[:, [0, 1, 3, 4, 5, 6, 8, 9]])

    # Edge cases
    empty_idx = torch.empty(0, dtype=torch.long)
    w_out_empty, w_reg_all = split_weight_channels(w, empty_idx, in_feature_dim=1)
    assert w_out_empty.shape == (8, 0)
    torch.testing.assert_close(w_reg_all, w)


def test_quantize_int8_vector_wise():
    """Verify row-wise vector INT8 quantization bounds and round-trip fidelity."""
    torch.manual_seed(42)
    x = torch.randn(8, 16) * 2.0
    q, scales = quantize_int8_vector_wise(x, qmax=127)

    assert q.dtype == torch.int8
    assert q.shape == (8, 16)
    assert scales.shape == (8, 1)
    assert int(q.min().item()) >= -127
    assert int(q.max().item()) <= 127

    # Reconstruction check
    x_recon = dequantize_int8_vector_wise(q, scales, dtype=x.dtype)
    max_err = torch.max(torch.abs(x - x_recon)).item()
    # INT8 with 127 steps has maximum quantization step <= scale/2
    assert max_err <= torch.max(scales).item() * 0.55


def test_quantize_int8_tensor_wise():
    """Verify tensor-wise INT8 quantization."""
    x = torch.tensor([[-5.0, 0.0, 5.0], [2.5, -2.5, 1.0]], dtype=torch.float32)
    q, scale = quantize_int8_tensor_wise(x, qmax=127)

    assert q.dtype == torch.int8
    assert int(q.min().item()) >= -127
    assert int(q.max().item()) <= 127
    assert scale.item() == pytest.approx(5.0 / 127.0, rel=1e-5)


def test_mixed_precision_matmul_fidelity_under_outliers():
    """
    Verify that mixed-precision activation matmul achieves superior SNR
    over uniform INT8 in the presence of activation outliers.
    """
    torch.manual_seed(42)
    batch_size = 16
    in_features = 128
    out_features = 64

    # Generate standard normal activations and weights
    x = torch.randn(batch_size, in_features) * 0.5
    w = torch.randn(out_features, in_features) / math.sqrt(in_features)

    # Inject severe outliers in 2 feature channels (typical in LLMs)
    x[:, 15] = 35.0
    x[:, 42] = -40.0

    stats = compare_mixed_vs_uniform_8bit(x, w, threshold=6.0, vector_wise=True)

    assert stats["num_outliers"] == 2
    # Pure INT8 should suffer catastrophic degradation under outliers
    # Mixed precision should have high SNR (> 30 dB) and high cosine similarity (> 0.999)
    assert stats["mixed_precision_snr_db"] > 30.0
    assert stats["mixed_precision_cosine_sim"] > 0.999
    # Mixed precision must outperform pure INT8 by a significant SNR margin (> 15 dB)
    assert stats["snr_improvement_db"] > 15.0


def test_mixed_precision_matmul_edge_cases():
    """Test zero outliers, all outliers, 3D tensor shapes, and biases."""
    torch.manual_seed(42)
    b, t, c, d = 2, 4, 16, 8
    x_3d = torch.randn(b, t, c)
    w = torch.randn(d, c)
    bias = torch.randn(d)

    # 1. No outliers (threshold very high)
    y_no_out = mixed_precision_matmul(x_3d, w, bias=bias, threshold=100.0)
    assert y_no_out.shape == (b, t, d)

    # 2. All outliers (threshold very low)
    y_all_out = mixed_precision_matmul(x_3d, w, bias=bias, threshold=0.01)
    assert y_all_out.shape == (b, t, d)
    # Since all outliers are in 16-bit float, output should closely match FP32
    y_ref = F.linear(x_3d, w, bias)
    torch.testing.assert_close(y_all_out, y_ref, atol=1e-2, rtol=1e-2)

    # 3. Shape mismatch check
    w_bad = torch.randn(d, c + 2)
    with pytest.raises(ValueError):
        mixed_precision_matmul(x_3d, w_bad)


def test_mixed_precision_ste_autograd():
    """Verify gradient backpropagation through mixed precision matmul using STE."""
    torch.manual_seed(42)
    x = torch.randn(4, 16, requires_grad=True)
    w = torch.randn(8, 16, requires_grad=True)
    bias = torch.randn(8, requires_grad=True)

    # Inject an outlier
    with torch.no_grad():
        x[0, 5] = 20.0

    y = mixed_precision_matmul_ste(x, w, bias=bias, threshold=6.0)
    loss = torch.sum(y ** 2)
    loss.backward()

    assert x.grad is not None
    assert w.grad is not None
    assert bias.grad is not None
    assert x.grad.shape == x.shape
    assert w.grad.shape == w.shape
    assert bias.grad.shape == bias.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(w.grad).any()
    assert not torch.isnan(bias.grad).any()


def test_mixed_precision_quantizer_oop():
    """Verify object-oriented MixedPrecisionActivationQuantizer."""
    torch.manual_seed(42)
    quantizer = MixedPrecisionActivationQuantizer(bits=8, threshold=6.0)

    x = torch.randn(6, 32) * 0.3
    x[:, 12] = 22.0
    x[:, 27] = -18.0

    state = quantizer.quantize(x)
    assert isinstance(state, MixedPrecisionState)
    assert state.outliers.shape == (6, 2)
    assert state.quantized_regular.shape == (6, 30)

    # Diagnostic metadata
    analysis = quantizer.analyze_outliers(x)
    assert analysis["num_outliers"] == 2
    assert analysis["total_channels"] == 32
    assert analysis["outlier_ratio"] == pytest.approx(2 / 32)

    # Dequantize roundtrip
    x_recon = quantizer.dequantize(state)
    assert x_recon.shape == x.shape
    # Outlier channels must be preserved with exact FP16 precision
    torch.testing.assert_close(x_recon[:, 12], x[:, 12].to(torch.float16).to(x.dtype), atol=1e-2, rtol=1e-2)

    # Matmul method
    w = torch.randn(16, 32)
    y = quantizer.matmul(x, w)
    assert y.shape == (6, 16)


def test_mixed_precision_linear_module():
    """Verify drop-in MixedPrecisionLinear PyTorch module."""
    torch.manual_seed(42)
    linear = nn.Linear(32, 16, bias=True)
    mp_linear = MixedPrecisionLinear.from_linear(linear, threshold=6.0)

    assert mp_linear.in_features == 32
    assert mp_linear.out_features == 16
    assert mp_linear.weight.shape == (16, 32)

    x = torch.randn(4, 32) * 0.4
    x[:, 5] = 25.0

    # Eval forward pass
    mp_linear.eval()
    y_eval = mp_linear(x)
    assert y_eval.shape == (4, 16)

    # Static outlier caching
    mp_linear.set_static_outliers(torch.tensor([5], dtype=torch.long))
    assert mp_linear.static_outliers is not None
    y_static = mp_linear(x)
    assert y_static.shape == (4, 16)
    mp_linear.clear_static_outliers()
    assert mp_linear.static_outliers is None

    # Memory profiling
    mem_stats = mp_linear.memory_stats()
    assert mem_stats["weight_bytes"] == 16 * 32 * 4
    assert mem_stats["compression_ratio"] > 1.0

    # Extra representation string
    repr_str = mp_linear.extra_repr()
    assert "in_features=32" in repr_str
    assert "threshold=6.0" in repr_str


def test_mixed_precision_dtypes():
    """Verify execution across different tensor data types."""
    torch.manual_seed(42)
    w = torch.randn(8, 16)
    x = torch.randn(4, 16)
    x[:, 3] = 15.0

    # Float32
    y32 = mixed_precision_matmul(x.float(), w.float(), threshold=6.0)
    assert y32.dtype == torch.float32

    # Float16
    y16 = mixed_precision_matmul(x.half(), w.half(), threshold=6.0, outlier_dtype=torch.float16)
    assert y16.dtype == torch.float16

    # Bfloat16
    ybf = mixed_precision_matmul(x.bfloat16(), w.bfloat16(), threshold=6.0, outlier_dtype=torch.bfloat16)
    assert ybf.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on host system")
def test_cuda_hardware_execution():
    """Verify hardware execution on NVIDIA GPU if CUDA is available."""
    device = torch.device("cuda")
    x = (torch.randn(8, 64, device=device) * 0.4).requires_grad_(True)
    w = (torch.randn(32, 64, device=device) * 0.2).requires_grad_(True)
    bias = torch.randn(32, device=device, requires_grad=True)

    with torch.no_grad():
        x[0, 10] = 30.0

    # Test CUDA STE matmul
    y = mixed_precision_matmul_ste(x, w, bias=bias, threshold=6.0)
    assert y.is_cuda
    loss = y.sum()
    loss.backward()

    assert x.grad.is_cuda
    assert w.grad.is_cuda

    # Test CUDA Linear layer
    linear = MixedPrecisionLinear(64, 32, device=device)
    y_layer = linear(x.detach())
    assert y_layer.is_cuda
