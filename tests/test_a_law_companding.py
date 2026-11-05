import math
import pytest
import torch

from inwhale.core.non_uniform import ALawQuantizer
from inwhale.core.uniform import SymmetricUniformQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


def make_quant(bits=8, A=87.6):
    obs = MinMaxObserver()
    rnd = NearestRounding()
    return ALawQuantizer(bits=bits, observer=obs, rounding=rnd, A=A)


def test_bits_boundaries():
    q4 = make_quant(bits=4)
    assert q4.qmin == -(1 << (4 - 1))
    assert q4.qmax == (1 << (4 - 1)) - 1

    q8 = make_quant(bits=8)
    assert q8.qmin == -128
    assert q8.qmax == 127

    q16 = make_quant(bits=16)
    assert q16.qmin == -32768
    assert q16.qmax == 32767


def test_zero_is_preserved():
    x = torch.zeros(50)
    q = make_quant(bits=8)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert torch.all(qx == 0)
    assert torch.all(dx == 0.0)


def test_quantized_values_within_bounds():
    x = torch.linspace(-25.0, 25.0, steps=500)
    q = make_quant(bits=8)

    qx = q.quantize(x)
    assert torch.all(qx >= q.qmin)
    assert torch.all(qx <= q.qmax)


def test_sign_preservation():
    x = torch.tensor([-12.5, -0.02, 0.0, 0.02, 12.5])
    q = make_quant(bits=8)

    qx = q.quantize(x)
    assert torch.all(torch.sign(qx) == torch.sign(x))


def test_monotonicity():
    x = torch.linspace(-10.0, 10.0, steps=250)
    q = make_quant(bits=8)

    qx = q.quantize(x)
    # Monotonically non-decreasing
    assert torch.all(qx[1:] >= qx[:-1])


def test_small_amplitude_resolution_advantage():
    """
    Core companding thesis: A-law companding allocates substantially higher
    bit resolution (smaller effective quantization steps) to low-amplitude
    signals near zero than linear uniform quantization of identical bit-width.
    """
    low_amp_x = torch.tensor([0.002, 0.005, 0.01, -0.002, -0.005, -0.01])
    # Full dynamic range reference including peaks
    full_signal = torch.cat([low_amp_x, torch.tensor([1.0, -1.0])])

    # 1. A-law quantizer (8 bits)
    q_alaw = make_quant(bits=8, A=87.6)
    qx_alaw = q_alaw.quantize(full_signal)
    dx_alaw = q_alaw.dequantize(qx_alaw)
    alaw_errors = (full_signal[: len(low_amp_x)] - dx_alaw[: len(low_amp_x)]).abs()

    # 2. Linear uniform quantizer (8 bits)
    obs_lin = MinMaxObserver()
    rnd_lin = NearestRounding()
    q_linear = SymmetricUniformQuantizer(bits=8, observer=obs_lin, rounding=rnd_lin)
    qx_lin = q_linear.quantize(full_signal)
    dx_lin = q_linear.dequantize(qx_lin)
    linear_errors = (full_signal[: len(low_amp_x)] - dx_lin[: len(low_amp_x)]).abs()

    # A-law should achieve substantially lower quantization error on small signals
    assert alaw_errors.mean() < linear_errors.mean()


def test_linear_region_exact_scaling():
    """
    Verify that in the linear region |x| < 1/A, the unquantized compression
    slope exactly equals A / (1 + ln(A)).
    """
    A = 87.6
    inv_A = 1.0 / A
    expected_slope = A / (1.0 + math.log(A))

    # Test points strictly within linear segment
    x_sub = torch.tensor([0.001, 0.003, 0.007, -0.002, -0.006])
    full_x = torch.cat([x_sub, torch.tensor([1.0, -1.0])])

    q = make_quant(bits=16, A=A)  # high bits to isolate quantization step
    qx = q.quantize(full_x)

    # In high bit-depth, quantized code ratio to qmax reflects compressed domain y
    y_approx = qx[: len(x_sub)].float() / q.qmax
    slopes = (y_approx / x_sub).abs()

    # Slopes should match expected continuous slope closely
    assert torch.allclose(slopes, torch.tensor(expected_slope, dtype=slopes.dtype), rtol=0.05)


def test_identity_at_A_equals_one():
    """
    When A = 1.0, A-law compression reduces to the linear identity mapping F(x) = x.
    """
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q_alaw1 = make_quant(bits=8, A=1.0)
    qx = q_alaw1.quantize(x)
    dx = q_alaw1.dequantize(qx)

    assert torch.isclose(dx, x, atol=0.02).all()


def test_peak_and_scale_reconstruction():
    x = torch.tensor([-12.0, -3.0, 0.0, 3.0, 12.0])
    q = make_quant(bits=8)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert q.scale is not None
    assert torch.isclose(dx[0], x[0], atol=0.2)
    assert torch.isclose(dx[-1], x[-1], atol=0.2)


def test_multidimensional_tensor_shapes():
    # 4D tensor (e.g. batch x channels x height x width)
    x = torch.randn(2, 3, 16, 16)
    q = make_quant(bits=8)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert qx.shape == x.shape
    assert dx.shape == x.shape


def test_invalid_parameters():
    obs = MinMaxObserver()
    rnd = NearestRounding()

    with pytest.raises(ValueError, match="A must be at least 1.0"):
        ALawQuantizer(bits=8, observer=obs, rounding=rnd, A=0.5)

    with pytest.raises(ValueError, match="A must be at least 1.0"):
        ALawQuantizer(bits=8, observer=obs, rounding=rnd, A=-10.0)

    with pytest.raises(ValueError, match="Bits must be at least 2"):
        ALawQuantizer(bits=1, observer=obs, rounding=rnd, A=87.6)


def test_dequantize_uninitialized_error():
    q = make_quant(bits=8)
    dummy_qx = torch.tensor([0, 10, -10])
    with pytest.raises(RuntimeError, match="Cannot dequantize before quantize"):
        q.dequantize(dummy_qx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution():
    x = torch.tensor([1.5, -0.8, 0.0, 3.2, -3.2], device="cuda")
    q = make_quant(bits=8)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert qx.is_cuda
    assert dx.is_cuda
    assert torch.all(qx >= q.qmin)
    assert torch.all(qx <= q.qmax)
