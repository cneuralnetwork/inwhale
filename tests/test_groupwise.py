import pytest
import torch
import torch.nn as nn

from inwhale.llm.groupwise import (
    GroupwiseQuantizer,
    GroupwiseLinear,
    groupwise_quantize,
    groupwise_dequantize,
    groupwise_fake_quantize,
)
from inwhale.rounding.nearest import NearestRounding


def test_groupwise_initialization_and_validation():
    """Verify parameter bounds and argument validation."""
    # Valid setup
    quantizer = GroupwiseQuantizer(bits=4, group_size=64, dim=-1, symmetric=True)
    assert quantizer.bits == 4
    assert quantizer.group_size == 64
    assert quantizer.dim == -1
    assert quantizer.symmetric is True
    assert quantizer.qmin == -8
    assert quantizer.qmax == 7

    # Asymmetric setup
    asym_quantizer = GroupwiseQuantizer(bits=4, group_size=64, dim=-1, symmetric=False)
    assert asym_quantizer.qmin == 0
    assert asym_quantizer.qmax == 15

    # Invalid bits
    with pytest.raises(ValueError, match="Bits must be >= 1"):
        GroupwiseQuantizer(bits=0)

    # Invalid group_size
    with pytest.raises(ValueError, match="Group size must be >= 1"):
        GroupwiseQuantizer(bits=4, group_size=0)

    # Invalid epsilon
    with pytest.raises(ValueError, match="Epsilon must be positive"):
        GroupwiseQuantizer(bits=4, group_size=32, eps=-1e-5)

    # Out-of-bounds dim on execution
    x = torch.randn(4, 8)
    with pytest.raises(IndexError, match="Dimension out of range"):
        groupwise_quantize(x, bits=4, group_size=4, dim=5)


def test_symmetric_quantization_codes_and_grid():
    """Verify symmetric integer codes strictly fall into [-2^(b-1), 2^(b-1) - 1]."""
    torch.manual_seed(42)
    x = torch.randn(16, 256)

    # 4-bit symmetric
    quantizer_4bit = GroupwiseQuantizer(bits=4, group_size=32, symmetric=True)
    qx = quantizer_4bit.quantize(x)
    assert qx.shape == x.shape
    assert qx.min() >= -8
    assert qx.max() <= 7
    # Verify integer discrete values
    assert torch.all(qx == torch.round(qx))
    assert quantizer_4bit.zero_point is None
    assert quantizer_4bit.scale is not None
    assert torch.all(quantizer_4bit.scale > 0)

    # Dequantize and check reconstruction error
    x_rec = quantizer_4bit.dequantize(qx)
    assert x_rec.shape == x.shape
    mse = torch.mean((x - x_rec) ** 2)
    assert mse < 0.05

    # 8-bit symmetric
    quantizer_8bit = GroupwiseQuantizer(bits=8, group_size=32, symmetric=True)
    qx8 = quantizer_8bit.quantize(x)
    assert qx8.min() >= -128
    assert qx8.max() <= 127
    x_rec8 = quantizer_8bit.dequantize(qx8)
    mse8 = torch.mean((x - x_rec8) ** 2)
    # 8-bit should have much lower MSE than 4-bit
    assert mse8 < mse / 100.0


def test_asymmetric_quantization_codes_and_zero_points():
    """Verify asymmetric quantization uses unsigned range [0, 2^b - 1] and valid zero-points."""
    torch.manual_seed(42)
    # Strictly non-centered tensor (e.g. ReLU activations or biased weights)
    x = torch.rand(8, 128) * 10.0 + 2.0

    quantizer = GroupwiseQuantizer(bits=4, group_size=32, symmetric=False)
    qx = quantizer.quantize(x)

    assert qx.shape == x.shape
    assert qx.min() >= 0
    assert qx.max() <= 15
    assert torch.all(qx == torch.round(qx))

    assert quantizer.zero_point is not None

    x_rec = quantizer.dequantize(qx)
    assert x_rec.shape == x.shape
    mse = torch.mean((x - x_rec) ** 2)
    assert mse < 0.05


def test_outlier_isolation():
    """
    Demonstrate that an outlier in Group 0 does NOT pollute the precision of Group 1.
    This is the core theoretical benefit of group-wise quantization in LLMs.
    """
    group_size = 64
    num_groups = 2
    x = torch.zeros(1, num_groups * group_size)

    # Group 0: normal values + 1 giant outlier (100.0)
    x[0, :group_size] = torch.randn(group_size) * 0.1
    x[0, 5] = 100.0

    # Group 1: pristine small values
    x[0, group_size:] = torch.randn(group_size) * 0.05

    # 1. Group-wise Quantization (G = 64)
    qx_gw, scale_gw, _, _ = groupwise_quantize(x, bits=4, group_size=group_size, symmetric=True)
    rec_gw = groupwise_dequantize(qx_gw, scale_gw, group_size=group_size)

    # Group 0 scale should be dominated by 100.0 / 7
    assert scale_gw[0, 0, 0] > 10.0
    # Group 1 scale should remain small (< 0.1)
    assert scale_gw[0, 1, 0] < 0.1

    err_group1_gw = torch.mean((x[0, group_size:] - rec_gw[0, group_size:]) ** 2).item()

    # 2. Per-Tensor / Full-Channel Quantization (G = 128, entire row in one group)
    qx_full, scale_full, _, _ = groupwise_quantize(x, bits=4, group_size=128, symmetric=True)
    rec_full = groupwise_dequantize(qx_full, scale_full, group_size=128)

    err_group1_full = torch.mean((x[0, group_size:] - rec_full[0, group_size:]) ** 2).item()

    # Group-wise error in Group 1 must be drastically lower (at least 20x lower)
    assert err_group1_gw < err_group1_full / 20.0


def test_non_divisible_dimension_padding():
    """Verify handling when tensor dimension length is not divisible by group_size."""
    torch.manual_seed(42)
    # Length 150 with group_size 64 -> requires 42 elements padding to reach 192 (3 groups)
    x = torch.randn(10, 150)
    group_size = 64

    qx, scale, zp, pad_len = groupwise_quantize(x, bits=4, group_size=group_size, dim=-1)
    assert pad_len == 42
    assert qx.shape == (10, 150)
    assert scale.shape == (10, 3, 1)

    rec = groupwise_dequantize(qx, scale, zero_point=zp, group_size=group_size, dim=-1, pad_len=pad_len)
    assert rec.shape == (10, 150)

    mse = torch.mean((x - rec) ** 2)
    assert mse < 0.05


def test_multidimensional_tensors_and_dim_selection():
    """Verify group-wise quantization across 1D, 2D, 3D, and 4D tensors with various dim choices."""
    torch.manual_seed(42)

    # 1D tensor
    x_1d = torch.randn(256)
    rec_1d = groupwise_fake_quantize(x_1d, bits=4, group_size=32, dim=0)
    assert rec_1d.shape == x_1d.shape
    assert torch.mean((x_1d - rec_1d) ** 2) < 0.05

    # 2D tensor along dim=0 (per-row grouping)
    x_2d = torch.randn(128, 64)
    rec_2d_dim0 = groupwise_fake_quantize(x_2d, bits=4, group_size=32, dim=0)
    assert rec_2d_dim0.shape == x_2d.shape

    # 3D tensor along dim=1
    x_3d = torch.randn(4, 64, 32)
    rec_3d = groupwise_fake_quantize(x_3d, bits=4, group_size=16, dim=1)
    assert rec_3d.shape == x_3d.shape

    # 4D convolutional weight tensor (out_channels, in_channels, H, W)
    x_4d = torch.randn(16, 32, 3, 3)
    rec_4d = groupwise_fake_quantize(x_4d, bits=4, group_size=16, dim=1)
    assert rec_4d.shape == x_4d.shape


def test_functional_and_class_api_parity():
    """Verify class-based quantizer produces identical results to functional helpers."""
    torch.manual_seed(42)
    x = torch.randn(8, 128)

    quantizer = GroupwiseQuantizer(bits=4, group_size=32, dim=-1, symmetric=True)
    qx_class = quantizer.quantize(x)
    rec_class = quantizer.dequantize(qx_class)

    qx_func, scale, zp, pad_len = groupwise_quantize(x, bits=4, group_size=32, dim=-1, symmetric=True)
    rec_func = groupwise_dequantize(qx_func, scale, zero_point=zp, group_size=32, dim=-1, pad_len=pad_len)

    fake_class = quantizer.fake_quantize(x)
    fake_func = groupwise_fake_quantize(x, bits=4, group_size=32, dim=-1, symmetric=True)

    assert torch.equal(qx_class, qx_func)
    assert torch.allclose(rec_class, rec_func)
    assert torch.allclose(fake_class, fake_func)


def test_groupwise_linear_layer():
    """Verify GroupwiseLinear drop-in module behavior and conversion from nn.Linear."""
    torch.manual_seed(42)
    in_features = 256
    out_features = 128
    batch_size = 16

    # 1. Direct initialization
    gw_layer = GroupwiseLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        bits=4,
        group_size=64,
        symmetric=True,
    )
    x = torch.randn(batch_size, in_features)
    out = gw_layer(x)
    assert out.shape == (batch_size, out_features)

    # 2. Conversion from standard nn.Linear
    linear = nn.Linear(in_features, out_features, bias=True)
    linear_out = linear(x)

    gw_from_linear = GroupwiseLinear.from_linear(
        linear=linear,
        bits=8,
        group_size=64,
        symmetric=True,
    )
    gw_out = gw_from_linear(x)

    assert gw_out.shape == linear_out.shape
    # 8-bit quantized linear should closely match full-precision output
    cos_sim = torch.nn.functional.cosine_similarity(linear_out.flatten(), gw_out.flatten(), dim=0)
    assert cos_sim.item() > 0.999

    # 3. Memory savings report
    savings = gw_from_linear.memory_savings()
    assert savings["num_elements"] == in_features * out_features
    assert savings["bits"] == 8
    assert savings["compression_ratio"] > 3.5


def test_edge_cases():
    """Verify behavior on empty tensors, all-zero tensors, and constant tensors."""
    # Empty tensor
    empty = torch.empty(0)
    rec_empty = groupwise_fake_quantize(empty, bits=4, group_size=32)
    assert rec_empty.numel() == 0

    # All-zero tensor
    zeros = torch.zeros(4, 64)
    rec_zeros = groupwise_fake_quantize(zeros, bits=4, group_size=32)
    assert torch.all(rec_zeros == 0.0)

    # Constant non-zero tensor
    constants = torch.full((4, 64), 7.5)
    rec_const = groupwise_fake_quantize(constants, bits=4, group_size=32)
    assert torch.allclose(rec_const, constants, atol=1e-3)

    # Group size larger than tensor length
    short = torch.randn(2, 10)
    rec_short = groupwise_fake_quantize(short, bits=4, group_size=64)
    assert rec_short.shape == short.shape
    assert torch.mean((short - rec_short) ** 2) < 0.05

    # Group size = 1 (extreme fine-grained)
    rec_g1 = groupwise_fake_quantize(short, bits=4, group_size=1)
    assert rec_g1.shape == short.shape

    # Dequantize before quantize raises error
    q = GroupwiseQuantizer(bits=4, group_size=32)
    with pytest.raises(RuntimeError, match="Call quantize\\(\\) first"):
        q.dequantize(short)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution():
    """Verify group-wise quantization operations and layers execute seamlessly on CUDA."""
    device = torch.device("cuda")
    x = torch.randn(32, 512, device=device)

    # Test quantizer on CUDA
    quantizer = GroupwiseQuantizer(bits=4, group_size=128, dim=-1)
    qx = quantizer.quantize(x)
    assert qx.is_cuda
    assert quantizer.scale.is_cuda

    x_rec = quantizer.dequantize(qx)
    assert x_rec.is_cuda
    assert x_rec.shape == x.shape

    # Test GroupwiseLinear on CUDA
    linear = nn.Linear(512, 256).to(device)
    gw_linear = GroupwiseLinear.from_linear(linear, bits=4, group_size=128)
    out = gw_linear(x)
    assert out.is_cuda
    assert out.shape == (32, 256)
