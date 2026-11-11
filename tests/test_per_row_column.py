"""
Unit tests for Per-Row and Per-Column Quantization.

Category 7: LLM-Specific Quantization (Milestone 7).
"""

import pytest
import torch
import torch.nn as nn

from inwhale.llm.per_row_column import (
    per_row_quantize,
    per_row_dequantize,
    per_row_fake_quantize,
    per_column_quantize,
    per_column_dequantize,
    per_column_fake_quantize,
    per_axis_quantize,
    per_axis_dequantize,
    per_axis_fake_quantize,
    vectorwise_quantized_matmul,
    PerRowQuantizer,
    PerColumnQuantizer,
    PerAxisQuantizer,
    PerRowLinear,
    PerColumnLinear,
)
from inwhale.rounding.bankers import BankersRounding


def test_per_row_quantizer_init_and_validation():
    """Verify initialization, negative bit errors, and parameter validation."""
    with pytest.raises(ValueError, match="Bits must be positive|Bits must be >= 1"):
        per_row_quantize(torch.randn(4, 4), bits=0)

    with pytest.raises(ValueError, match="Bits must be positive|Bits must be >= 1"):
        PerRowQuantizer(bits=-1)

    quantizer = PerRowQuantizer(bits=8, symmetric=True)
    assert quantizer.bits == 8
    assert quantizer.symmetric is True
    assert quantizer.scale is None

    # Dequantize before quantize should raise
    with pytest.raises(RuntimeError, match="Cannot dequantize before quantize"):
        quantizer.dequantize(torch.zeros(4, 4))


def test_per_row_symmetric_integer_grid():
    """Verify symmetric per-row INT8 and INT4 quantization grid constraints."""
    torch.manual_seed(42)
    x = torch.randn(8, 32) * 5.0

    # INT8 test: range [-128, 127]
    qx8, s8, zp8 = per_row_quantize(x, bits=8, symmetric=True)
    assert zp8 is None
    assert s8.shape == (8, 1)
    assert qx8.shape == (8, 32)
    assert qx8.min() >= -128
    assert qx8.max() <= 127
    assert torch.all(qx8 == torch.round(qx8))

    x_rec8 = per_row_dequantize(qx8, s8)
    mse8 = torch.mean((x - x_rec8) ** 2).item()
    assert mse8 < 0.01

    # INT4 test: range [-8, 7]
    qx4, s4, zp4 = per_row_quantize(x, bits=4, symmetric=True)
    assert zp4 is None
    assert s4.shape == (8, 1)
    assert qx4.min() >= -8
    assert qx4.max() <= 7
    assert torch.all(qx4 == torch.round(qx4))


def test_per_row_asymmetric_grid():
    """Verify asymmetric per-row quantization with zero-point offsets."""
    torch.manual_seed(42)
    # Asymmetric positive distribution like ReLU activations [0.0, 10.0]
    x = torch.rand(6, 40) * 10.0 + 2.0

    qx, s, zp = per_row_quantize(x, bits=8, symmetric=False, signed=False)
    assert zp is not None
    assert s.shape == (6, 1)
    assert zp.shape == (6, 1)
    assert qx.shape == (6, 40)
    assert qx.min() >= 0
    assert qx.max() <= 255
    assert zp.min() >= 0
    assert zp.max() <= 255

    x_rec = per_row_dequantize(qx, s, zp)
    mse = torch.mean((x - x_rec) ** 2).item()
    assert mse < 0.005


def test_per_column_symmetric_and_asymmetric_grid():
    """Verify per-column quantization scale shapes and numerical grid."""
    torch.manual_seed(42)
    x = torch.randn(16, 8) * 3.0

    # Symmetric per-column: scale shape must be (1, 8)
    qx_sym, s_sym, zp_sym = per_column_quantize(x, bits=8, symmetric=True)
    assert zp_sym is None
    assert s_sym.shape == (1, 8)
    assert qx_sym.shape == (16, 8)
    assert qx_sym.min() >= -128
    assert qx_sym.max() <= 127

    x_rec_sym = per_column_dequantize(qx_sym, s_sym)
    mse_sym = torch.mean((x - x_rec_sym) ** 2).item()
    assert mse_sym < 0.01

    # Asymmetric per-column
    qx_asym, s_asym, zp_asym = per_column_quantize(x, bits=8, symmetric=False)
    assert zp_asym is not None
    assert s_asym.shape == (1, 8)
    assert zp_asym.shape == (1, 8)
    assert qx_asym.min() >= 0
    assert qx_asym.max() <= 255

    x_rec_asym = per_column_dequantize(qx_asym, s_asym, zp_asym)
    mse_asym = torch.mean((x - x_rec_asym) ** 2).item()
    assert mse_asym < 0.01


def test_row_outlier_isolation():
    """
    Mathematical verification that an extreme outlier in Row 0 does NOT degrade Row 1.
    Per-row quantization isolates row variations, reducing Row 1 MSE by orders of magnitude.
    """
    torch.manual_seed(123)
    row0 = torch.randn(1, 64) * 0.1
    row0[0, 0] = 50.0  # Massive outlier in Row 0

    row1 = torch.randn(1, 64) * 0.1  # Normal small-magnitude distribution

    matrix = torch.cat([row0, row1], dim=0)  # Shape (2, 64)

    # 1. Coarse per-tensor quantization
    max_tensor = torch.max(torch.abs(matrix))
    s_tensor = max_tensor / 127.0
    q_tensor = torch.clamp(torch.round(matrix / s_tensor), -128, 127)
    rec_tensor = q_tensor * s_tensor
    row1_mse_tensor = torch.mean((row1 - rec_tensor[1:]) ** 2).item()

    # 2. Per-row quantization
    qx_row, s_row, _ = per_row_quantize(matrix, bits=8, symmetric=True)
    rec_row = per_row_dequantize(qx_row, s_row)
    row1_mse_row = torch.mean((row1 - rec_row[1:]) ** 2).item()

    # Scale for row1 in per-row quantization is tiny (~0.1/127), whereas per-tensor scale is ~50/127
    assert s_row[1, 0] < s_tensor / 50.0
    # Row 1 error under per-row is dramatically lower (>50x improvement)
    assert row1_mse_row < row1_mse_tensor / 50.0


def test_column_outlier_isolation():
    """
    Mathematical verification that an extreme outlier in Column 0 does NOT degrade Column 1.
    """
    torch.manual_seed(123)
    col0 = torch.randn(64, 1) * 0.1
    col0[0, 0] = 50.0  # Massive outlier in Column 0

    col1 = torch.randn(64, 1) * 0.1  # Normal small-magnitude distribution

    matrix = torch.cat([col0, col1], dim=1)  # Shape (64, 2)

    # 1. Coarse per-tensor quantization
    s_tensor = torch.max(torch.abs(matrix)) / 127.0
    q_tensor = torch.clamp(torch.round(matrix / s_tensor), -128, 127)
    rec_tensor = q_tensor * s_tensor
    col1_mse_tensor = torch.mean((col1 - rec_tensor[:, 1:]) ** 2).item()

    # 2. Per-column quantization
    qx_col, s_col, _ = per_column_quantize(matrix, bits=8, symmetric=True)
    rec_col = per_column_dequantize(qx_col, s_col)
    col1_mse_col = torch.mean((col1 - rec_col[:, 1:]) ** 2).item()

    assert s_col[0, 1] < s_tensor / 50.0
    assert col1_mse_col < col1_mse_tensor / 50.0


def test_vectorwise_quantized_matmul():
    """
    Verify Vector-wise INT8 GEMM (LLM.int8() style):
    Per-row activation quantization combined with per-column weight quantization.
    """
    torch.manual_seed(99)
    # Activations (Tokens M=16, Hidden K=128)
    activation = torch.randn(16, 128)
    # Weight (Hidden K=128, Out N=64)
    weight = torch.randn(128, 64)

    # Standard FP32 GEMM reference
    c_ref = torch.matmul(activation, weight)

    # Vector-wise INT8 GEMM
    c_quant = vectorwise_quantized_matmul(activation, weight, bits=8)

    assert c_quant.shape == (16, 64)
    # Cosine similarity must be near 1.0 (>0.999)
    cosine_sim = torch.cosine_similarity(
        c_ref.flatten(), c_quant.flatten(), dim=0
    ).item()
    assert cosine_sim > 0.999

    # Relative MSE should be small (<0.01)
    rel_mse = (torch.mean((c_ref - c_quant) ** 2) / torch.var(c_ref)).item()
    assert rel_mse < 0.005

    # Batched activation (Batch B=2, Tokens S=8, Hidden K=128)
    act_batched = torch.randn(2, 8, 128)
    c_batched_ref = torch.matmul(act_batched, weight)
    c_batched_quant = vectorwise_quantized_matmul(act_batched, weight, bits=8)
    assert c_batched_quant.shape == (2, 8, 64)
    batched_cos_sim = torch.cosine_similarity(
        c_batched_ref.flatten(), c_batched_quant.flatten(), dim=0
    ).item()
    assert batched_cos_sim > 0.999

    # Transposed weight support (Out N=64, Hidden K=128)
    weight_t = weight.t()
    c_transposed = vectorwise_quantized_matmul(
        activation, weight_t, bits=8, transpose_weight=True
    )
    assert torch.allclose(c_quant, c_transposed, atol=1e-5)


def test_multidimensional_tensors_3d_4d():
    """Verify per-row and per-column quantization on 3D and 4D tensors."""
    torch.manual_seed(42)
    # 3D: (Batch=4, Rows=8, Cols=16)
    x3d = torch.randn(4, 8, 16)
    qx3d_row, s3d_row, _ = per_row_quantize(x3d, bits=8)
    assert s3d_row.shape == (4, 8, 1)
    rec3d_row = per_row_dequantize(qx3d_row, s3d_row)
    assert rec3d_row.shape == x3d.shape

    qx3d_col, s3d_col, _ = per_column_quantize(x3d, bits=8)
    assert s3d_col.shape == (4, 1, 16)
    rec3d_col = per_column_dequantize(qx3d_col, s3d_col)
    assert rec3d_col.shape == x3d.shape

    # 4D: (Batch=2, Heads=4, Seq=16, Dim=32)
    x4d = torch.randn(2, 4, 16, 32)
    qx4d_row, s4d_row, _ = per_row_quantize(x4d, bits=8)
    assert s4d_row.shape == (2, 4, 16, 1)
    qx4d_col, s4d_col, _ = per_column_quantize(x4d, bits=8)
    assert s4d_col.shape == (2, 4, 1, 32)


def test_per_axis_dispatcher():
    """Verify per_axis_quantize with various axis string and integer specifications."""
    x = torch.randn(10, 20)

    # By string 'row'
    qx_r, s_r, _ = per_axis_quantize(x, axis="row", bits=8)
    assert s_r.shape == (10, 1)

    # By string 'column'
    qx_c, s_c, _ = per_axis_quantize(x, axis="column", bits=8)
    assert s_c.shape == (1, 20)

    # By integer axis for 2D
    qx_0, s_0, _ = per_axis_quantize(x, axis=0, bits=8)
    assert s_0.shape == (10, 1)

    qx_1, s_1, _ = per_axis_quantize(x, axis=1, bits=8)
    assert s_1.shape == (1, 20)

    # Test fake quantization and dequantization helpers
    fake_r = per_row_fake_quantize(x, bits=8)
    assert fake_r.shape == x.shape
    fake_c = per_column_fake_quantize(x, bits=8)
    assert fake_c.shape == x.shape
    fake_a = per_axis_fake_quantize(x, axis="row", bits=8)
    assert torch.allclose(fake_r, fake_a)

    deq_a = per_axis_dequantize(qx_r, s_r)
    assert torch.allclose(deq_a, fake_r)

    with pytest.raises(ValueError, match="Unrecognized axis string"):
        per_axis_quantize(x, axis="diagonal")


def test_classes_quantize_dequantize():
    """Verify PerRowQuantizer, PerColumnQuantizer, and PerAxisQuantizer classes."""
    x = torch.randn(12, 24)

    # PerRowQuantizer
    row_q = PerRowQuantizer(bits=8, symmetric=True)
    qx_r = row_q.quantize(x)
    assert row_q.scale.shape == (12, 1)
    rec_r = row_q.dequantize(qx_r)
    assert rec_r.shape == x.shape

    # PerColumnQuantizer
    col_q = PerColumnQuantizer(bits=8, symmetric=False)
    qx_c = col_q.quantize(x)
    assert col_q.scale.shape == (1, 24)
    assert col_q.zero_point.shape == (1, 24)
    rec_c = col_q.dequantize(qx_c)
    assert rec_c.shape == x.shape

    # PerAxisQuantizer
    axis_q = PerAxisQuantizer(bits=8, axis="row")
    qx_a = axis_q.quantize(x)
    rec_a = axis_q.dequantize(qx_a)
    assert torch.allclose(rec_r, rec_a)


def test_nn_linear_layers():
    """Verify PerRowLinear and PerColumnLinear modules."""
    torch.manual_seed(42)
    in_features, out_features = 64, 32
    linear = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(8, in_features)

    ref_out = linear(x)

    # PerRowLinear
    row_layer = PerRowLinear.from_linear(linear, bits=8, symmetric=True)
    assert row_layer.scales.shape == (out_features, 1)
    row_out = row_layer(x)
    assert row_out.shape == (8, out_features)
    cos_sim_row = torch.cosine_similarity(
        ref_out.flatten(), row_out.flatten(), dim=0
    ).item()
    assert cos_sim_row > 0.999

    savings_row = row_layer.memory_savings()
    assert savings_row["bits"] == 8
    assert savings_row["compression_ratio"] > 3.0

    # PerColumnLinear
    col_layer = PerColumnLinear.from_linear(linear, bits=8, symmetric=True)
    assert col_layer.scales.shape == (1, in_features)
    col_out = col_layer(x)
    assert col_out.shape == (8, out_features)
    cos_sim_col = torch.cosine_similarity(
        ref_out.flatten(), col_out.flatten(), dim=0
    ).item()
    assert cos_sim_col > 0.999

    savings_col = col_layer.memory_savings()
    assert savings_col["compression_ratio"] > 3.0


def test_rounding_strategies_compatibility():
    """Verify that different RoundingStrategy implementations work correctly."""
    x = torch.randn(8, 16)
    bankers = BankersRounding()

    qx_bankers, s, _ = per_row_quantize(x, bits=8, rounding=bankers)
    assert qx_bankers.shape == x.shape
    assert s.shape == (8, 1)


def test_edge_cases():
    """Verify behavior on empty tensors, all-zeros, constants, and 1-bit binary."""
    # 1. Empty tensor
    empty = torch.empty(0, 5)
    qx_e, s_e, zp_e = per_row_quantize(empty, bits=8)
    assert qx_e.numel() == 0
    rec_e = per_row_dequantize(qx_e, s_e, zp_e)
    assert rec_e.numel() == 0

    # 2. All zeros tensor
    zeros = torch.zeros(4, 8)
    qx_z, s_z, _ = per_row_quantize(zeros, bits=8)
    assert torch.all(qx_z == 0)
    assert torch.all(s_z >= 1e-8)

    # 3. Constant tensor
    const = torch.full((4, 8), 3.5)
    qx_c, s_c, _ = per_column_quantize(const, bits=8)
    rec_c = per_column_dequantize(qx_c, s_c)
    assert torch.allclose(rec_c, const, atol=1e-3)

    # 4. 1-bit binary quantization
    b1_x = torch.tensor([[-2.0, 1.5, -0.5], [3.0, -4.0, 2.0]])
    qx_1b, s_1b, _ = per_row_quantize(b1_x, bits=1, symmetric=True)
    assert torch.all((qx_1b == 1.0) | (qx_1b == -1.0))
    assert s_1b.shape == (2, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution():
    """Verify execution on NVIDIA CUDA GPU hardware."""
    device = torch.device("cuda:0")
    x = torch.randn(32, 64, device=device)

    # Per-row on CUDA
    qx_r, s_r, _ = per_row_quantize(x, bits=8)
    assert qx_r.device.type == "cuda"
    assert s_r.device.type == "cuda"
    rec_r = per_row_dequantize(qx_r, s_r)
    assert rec_r.device.type == "cuda"

    # Per-column on CUDA
    qx_c, s_c, _ = per_column_quantize(x, bits=8)
    assert qx_c.device.type == "cuda"
    assert s_c.device.type == "cuda"

    # Vector-wise GEMM on CUDA
    w = torch.randn(64, 48, device=device)
    c_cuda = vectorwise_quantized_matmul(x, w, bits=8)
    assert c_cuda.device.type == "cuda"

    # PerRowLinear and PerColumnLinear on CUDA
    linear = nn.Linear(64, 48).to(device)
    layer = PerRowLinear.from_linear(linear, bits=8)
    out = layer(x)
    assert out.device.type == "cuda"
