"""
Unit tests for NF4 (NormalFloat4) and FP4 Quantization Formats.

Category 7: LLM-Specific Quantization (Milestone 7).
"""

import math
import pytest
import torch
import torch.nn as nn

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


def test_nf4_canonical_codebook_properties():
    """Verify NF4 canonical codebook length, bounds, exact zero, and monotonicity."""
    cb = torch.tensor(NF4_CANONICAL_CODEBOOK)
    assert cb.numel() == 16, "NF4 codebook must contain exactly 16 levels (4-bit)."
    assert cb.min().item() == pytest.approx(-1.0, abs=1e-6)
    assert cb.max().item() == pytest.approx(1.0, abs=1e-6)
    # Exact zero must be at index 7 (separating 7 negative and 8 positive levels)
    assert cb[7].item() == 0.0, "NF4 codebook must represent exact zero at index 7."
    # Strictly monotonically increasing
    diffs = cb[1:] - cb[:-1]
    assert (diffs > 0).all(), "NF4 codebook levels must be strictly monotonically increasing."


def test_create_normal_map_algorithmic():
    """Verify algorithmic normal map derivation matches canonical codebook within tight tolerance."""
    derived_cb = create_normal_map(num_bits=4)
    canonical_cb = torch.tensor(NF4_CANONICAL_CODEBOOK)
    assert derived_cb.shape == canonical_cb.shape
    max_diff = (derived_cb - canonical_cb).abs().max().item()
    assert max_diff < 1e-4, f"Algorithmic NF4 codebook differs by {max_diff}"

    with pytest.raises(ValueError, match="supports 4-bit"):
        create_normal_map(num_bits=8)


def test_fp4_codebooks():
    """Verify FP4 E2M1, BNB, and E3M0 codebook properties."""
    for fp4_type in [FP4Type.E2M1, FP4Type.BNB, FP4Type.E3M0]:
        cb = get_codebook("fp4", fp4_type=fp4_type)
        assert cb.numel() == 16
        assert cb.min().item() == pytest.approx(-1.0, abs=1e-6)
        assert cb.max().item() == pytest.approx(1.0, abs=1e-6)
        # Levels should be non-decreasing
        diffs = cb[1:] - cb[:-1]
        assert (diffs >= 0).all(), f"FP4 {fp4_type} levels must be non-decreasing."

    with pytest.raises(ValueError, match="Unknown format_name"):
        get_codebook("unknown_format")

    with pytest.raises(ValueError, match="is not a valid FP4Type|Unknown FP4 type"):
        get_codebook("fp4", fp4_type="invalid_type")


def test_pack_unpack_4bit_roundtrip():
    """Verify 4-bit nibble packing and unpacking round-trip across various tensor sizes."""
    test_sizes = [0, 1, 2, 7, 8, 63, 64, 127, 128, 512, 1025]
    for n in test_sizes:
        if n == 0:
            indices = torch.empty(0, dtype=torch.uint8)
        else:
            indices = torch.randint(0, 16, (n,), dtype=torch.uint8)

        packed = pack_4bit(indices)
        expected_packed_len = (n + 1) // 2
        assert packed.numel() == expected_packed_len

        unpacked = unpack_4bit(packed, original_numel=n)
        assert torch.equal(indices, unpacked), f"Failed pack/unpack round-trip for size {n}"

    # Multidimensional test
    orig_2d = torch.randint(0, 16, (16, 32), dtype=torch.uint8)
    packed_2d = pack_4bit(orig_2d)
    unpacked_2d = unpack_4bit(packed_2d, original_numel=orig_2d.numel(), original_shape=orig_2d.shape)
    assert torch.equal(orig_2d, unpacked_2d)

    # Test out-of-range value raises
    bad_indices = torch.tensor([0, 1, 16], dtype=torch.uint8)
    with pytest.raises(ValueError, match="must be in range \\[0, 15\\]"):
        pack_4bit(bad_indices)


def test_nf4_quantize_dequantize_blockwise():
    """Verify NF4 block-wise quantization and dequantization with different block sizes."""
    torch.manual_seed(42)
    x = torch.randn(128, 64)

    for block_size in [16, 32, 64]:
        # Unpacked mode
        indices, scales, pad_len, orig_shape = nf4_quantize(x, block_size=block_size, pack=False)
        assert indices.shape == x.shape
        assert (indices >= 0).all() and (indices <= 15).all()
        assert scales.shape == (x.numel() // block_size, 1)

        x_hat = nf4_dequantize(indices, scales, block_size=block_size, pad_len=pad_len, shape=orig_shape, is_packed=False)
        assert x_hat.shape == x.shape

        # Reconstructed tensor should have high cosine similarity with x
        cos_sim = torch.cosine_similarity(x.reshape(-1), x_hat.reshape(-1), dim=0).item()
        assert cos_sim > 0.99, f"NF4 cosine similarity too low ({cos_sim}) for block_size={block_size}"

        # Packed mode
        packed_indices, scales_p, pad_len_p, orig_shape_p = nf4_quantize(x, block_size=block_size, pack=True)
        assert packed_indices.numel() == x.numel() // 2
        x_hat_p = nf4_dequantize(packed_indices, scales_p, block_size=block_size, pad_len=pad_len_p, shape=orig_shape_p, is_packed=True)
        assert torch.allclose(x_hat, x_hat_p, atol=1e-6)


def test_fp4_quantize_dequantize_blockwise():
    """Verify FP4 quantization across E2M1, BNB, and E3M0 types."""
    torch.manual_seed(42)
    x = torch.randn(64, 64)

    for fp4_type in [FP4Type.E2M1, FP4Type.BNB, FP4Type.E3M0]:
        indices, scales, pad_len, orig_shape = fp4_quantize(x, block_size=32, fp4_type=fp4_type, pack=True)
        x_hat = fp4_dequantize(indices, scales, block_size=32, fp4_type=fp4_type, pad_len=pad_len, shape=orig_shape, is_packed=True)
        cos_sim = torch.cosine_similarity(x.reshape(-1), x_hat.reshape(-1), dim=0).item()
        assert cos_sim > 0.95, f"FP4 ({fp4_type}) cosine similarity too low: {cos_sim}"


def test_nf4_superiority_on_gaussian_weights():
    """
    Empirical validation of QLoRA theorem:
    NF4 achieves significantly lower MSE and higher SNR than uniform INT4 on normally distributed weights.
    """
    torch.manual_seed(123)
    # Generate standard normal weights N(0, 1) typical of deep learning linear layers
    weights = torch.randn(256, 256)

    # 1. NF4 Quantization (block_size=64)
    nf4_hat = nf4_fake_quantize(weights, block_size=64)
    mse_nf4 = torch.mean((weights - nf4_hat) ** 2).item()
    snr_nf4 = 10 * math.log10(torch.mean(weights ** 2).item() / mse_nf4)

    # 2. Uniform INT4 Quantization with block_size=64
    # INT4 has 16 uniform levels from -8 to 7 or -7 to 7
    blocks = weights.reshape(-1, 64)
    scales = torch.max(blocks.abs(), dim=-1, keepdim=True).values / 7.0
    int4_q = torch.clamp(torch.round(blocks / scales), -7, 7)
    int4_hat = (int4_q * scales).reshape(weights.shape)
    mse_int4 = torch.mean((weights - int4_hat) ** 2).item()
    snr_int4 = 10 * math.log10(torch.mean(weights ** 2).item() / mse_int4)

    # NF4 should have lower MSE and higher SNR than uniform INT4 on Gaussian data
    assert mse_nf4 < mse_int4, f"Expected NF4 MSE ({mse_nf4:.5f}) < INT4 MSE ({mse_int4:.5f})"
    assert snr_nf4 > snr_int4, f"Expected NF4 SNR ({snr_nf4:.2f} dB) > INT4 SNR ({snr_int4:.2f} dB)"
    snr_gain = snr_nf4 - snr_int4
    assert snr_gain > 0.5, f"Expected significant SNR gain for NF4, got +{snr_gain:.2f} dB"


def test_nf4_fp4_fake_quantize_shapes_and_dtypes():
    """Verify fake_quantize preserves arbitrary shapes and supports float32 and float16."""
    shapes = [(32,), (16, 32), (4, 8, 16), (2, 3, 4, 16)]
    for shape in shapes:
        x = torch.randn(*shape)
        x_nf4 = nf4_fake_quantize(x, block_size=16)
        assert x_nf4.shape == x.shape
        assert x_nf4.dtype == x.dtype

        x_fp4 = fp4_fake_quantize(x, block_size=16, fp4_type="e2m1")
        assert x_fp4.shape == x.shape
        assert x_fp4.dtype == x.dtype

    # Test Float16 and Bfloat16 support
    x_f16 = torch.randn(32, 32, dtype=torch.float16)
    out_f16 = nf4_fake_quantize(x_f16, block_size=32)
    assert out_f16.dtype == torch.float16

    x_bf16 = torch.randn(32, 32, dtype=torch.bfloat16)
    out_bf16 = fp4_fake_quantize(x_bf16, block_size=32)
    assert out_bf16.dtype == torch.bfloat16


def test_ste_autograd_backward():
    """Verify Straight-Through Estimator allows gradient flow during backward pass."""
    x = torch.randn(32, 32, requires_grad=True)
    out = nf4_fake_quantize(x, block_size=32, use_ste=True)
    loss = (out ** 2).sum()
    loss.backward()

    assert x.grad is not None
    # With STE, grad_x should be non-zero and directly proportional to 2 * out
    assert not torch.all(x.grad == 0)
    assert x.grad.shape == x.shape

    # Test FP4 STE
    x_fp4 = torch.randn(32, 32, requires_grad=True)
    out_fp4 = fp4_fake_quantize(x_fp4, block_size=32, use_ste=True)
    loss_fp4 = (out_fp4 ** 2).sum()
    loss_fp4.backward()
    assert x_fp4.grad is not None


def test_nf4_quantizer_oop_api():
    """Verify NF4Quantizer object-oriented BaseQuantizer class interface."""
    quantizer = NF4Quantizer(block_size=64, pack=True)
    assert quantizer.bits == 4
    assert quantizer.block_size == 64
    assert quantizer.scales is None

    # Dequantize before quantize must raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot dequantize before quantize"):
        quantizer.dequantize(torch.empty(0))

    x = torch.randn(128, 128)
    packed_indices = quantizer.quantize(x)
    assert quantizer.scales is not None
    assert packed_indices.dtype == torch.uint8
    assert packed_indices.numel() == x.numel() // 2

    x_hat = quantizer.dequantize(packed_indices)
    assert x_hat.shape == x.shape

    # Fake quantize
    x_fake = quantizer.fake_quantize(x)
    assert x_fake.shape == x.shape

    with pytest.raises(ValueError, match="block_size must be >= 1"):
        NF4Quantizer(block_size=0)


def test_fp4_quantizer_oop_api():
    """Verify FP4Quantizer object-oriented BaseQuantizer class interface."""
    quantizer = FP4Quantizer(block_size=32, fp4_type="e2m1", pack=True)
    assert quantizer.bits == 4
    assert quantizer.fp4_type == FP4Type.E2M1

    with pytest.raises(RuntimeError, match="Cannot dequantize before quantize"):
        quantizer.dequantize(torch.empty(0))

    x = torch.randn(64, 64)
    packed = quantizer.quantize(x)
    x_hat = quantizer.dequantize(packed)
    assert x_hat.shape == x.shape

    with pytest.raises(ValueError, match="block_size must be >= 1"):
        FP4Quantizer(block_size=-5)


def test_nf4_linear_layer():
    """Verify NF4Linear drop-in neural network module initialization and forward pass."""
    torch.manual_seed(42)
    in_features, out_features = 128, 64
    layer = NF4Linear(in_features, out_features, bias=True, block_size=64)

    assert layer.weight_packed.dtype == torch.uint8
    assert layer.weight_packed.numel() == (in_features * out_features) // 2
    assert layer.bias is not None

    # Test forward pass with batched input
    x = torch.randn(8, in_features)
    y = layer(x)
    assert y.shape == (8, out_features)

    # Test without bias
    layer_no_bias = NF4Linear(in_features, out_features, bias=False, block_size=64)
    assert layer_no_bias.bias is None
    y_no_bias = layer_no_bias(x)
    assert y_no_bias.shape == (8, out_features)


def test_fp4_linear_layer():
    """Verify FP4Linear drop-in neural network module."""
    torch.manual_seed(42)
    in_features, out_features = 64, 32
    layer = FP4Linear(in_features, out_features, bias=True, block_size=32, fp4_type="e2m1")
    assert layer.weight_packed.numel() == (in_features * out_features) // 2

    x = torch.randn(4, in_features)
    y = layer(x)
    assert y.shape == (4, out_features)


def test_linear_from_linear_and_memory_footprint():
    """Verify from_linear conversion factory and memory footprint analysis."""
    torch.manual_seed(42)
    in_dim, out_dim = 256, 128
    orig_linear = nn.Linear(in_dim, out_dim, bias=True)

    # Convert to NF4Linear
    nf4_layer = NF4Linear.from_linear(orig_linear, block_size=64)
    # Convert to FP4Linear
    fp4_layer = FP4Linear.from_linear(orig_linear, block_size=32, fp4_type="e2m1")

    # Evaluate outputs on same input
    x = torch.randn(16, in_dim)
    orig_out = orig_linear(x)
    nf4_out = nf4_layer(x)
    fp4_out = fp4_layer(x)

    # Cosine fidelity should be exceptionally high (> 0.99)
    cos_nf4 = torch.cosine_similarity(orig_out.reshape(-1), nf4_out.reshape(-1), dim=0).item()
    cos_fp4 = torch.cosine_similarity(orig_out.reshape(-1), fp4_out.reshape(-1), dim=0).item()
    assert cos_nf4 > 0.99, f"NF4 output cosine similarity {cos_nf4} is below 0.99"
    assert cos_fp4 > 0.98, f"FP4 output cosine similarity {cos_fp4} is below 0.98"

    # Verify memory footprint
    stats_nf4 = nf4_layer.memory_footprint()
    assert stats_nf4["num_parameters"] == in_dim * out_dim
    # Compressed bytes should be at least 6.8x smaller than FP32 baseline
    assert stats_nf4["compression_vs_fp32"] > 6.8, f"Compression ratio {stats_nf4['compression_vs_fp32']} < 6.8x"
    assert stats_nf4["compression_vs_fp16"] > 3.4

    stats_fp4 = fp4_layer.memory_footprint()
    assert stats_fp4["compression_vs_fp32"] > 6.0


def test_edge_cases():
    """Verify edge cases: all zeros, single-element, non-multiple block sizes, outliers."""
    # 1. All-zero tensor
    zeros = torch.zeros(64, 64)
    zeros_hat = nf4_fake_quantize(zeros, block_size=64)
    assert torch.all(zeros_hat == 0.0), "All-zero tensor must reconstruct to exact zeros."

    # 2. Non-multiple block size (requires padding and unpadding)
    odd_tensor = torch.randn(37, 19)  # 703 elements, not divisible by 64
    odd_hat = nf4_fake_quantize(odd_tensor, block_size=64)
    assert odd_hat.shape == odd_tensor.shape
    cos_sim = torch.cosine_similarity(odd_tensor.reshape(-1), odd_hat.reshape(-1), dim=0).item()
    assert cos_sim > 0.99

    # 3. Single element
    single = torch.tensor([3.14159])
    single_hat = nf4_fake_quantize(single, block_size=64)
    assert single_hat.shape == (1,)
    assert pytest.approx(single_hat.item(), abs=0.1) == single.item()

    # 4. Outlier value
    outlier_t = torch.randn(64, 64)
    outlier_t[0, 0] = 500.0  # extreme outlier
    outlier_hat = nf4_fake_quantize(outlier_t, block_size=64)
    assert outlier_hat.shape == outlier_t.shape
    assert not torch.isnan(outlier_hat).any()
    assert not torch.isinf(outlier_hat).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_cuda_hardware_execution():
    """Verify NF4 and FP4 operations execute correctly on CUDA GPU hardware."""
    device = torch.device("cuda")
    x = torch.randn(128, 128, device=device)

    # NF4 on CUDA
    nf4_out = nf4_fake_quantize(x, block_size=64)
    assert nf4_out.device.type == "cuda"
    assert nf4_out.shape == x.shape

    # FP4 on CUDA
    fp4_out = fp4_fake_quantize(x, block_size=32, fp4_type="e2m1")
    assert fp4_out.device.type == "cuda"

    # NF4Linear on CUDA
    linear = NF4Linear(128, 64, device=device)
    y = linear(x)
    assert y.device.type == "cuda"
    assert y.shape == (128, 64)
