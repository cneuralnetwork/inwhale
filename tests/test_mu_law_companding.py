import pytest
import torch

from inwhale.core.non_uniform import MuLawQuantizer
from inwhale.core.uniform import SymmetricUniformQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


def make_quant(bits=8, mu=255.0):
    obs = MinMaxObserver()
    rnd = NearestRounding()
    return MuLawQuantizer(bits=bits, observer=obs, rounding=rnd, mu=mu)


def test_bits_boundaries():
    q4 = make_quant(bits=4)
    assert q4.qmin == -(1 << (4 - 1))
    assert q4.qmax == (1 << (4 - 1)) - 1

    q8 = make_quant(bits=8)
    assert q8.qmin == -128
    assert q8.qmax == 127


def test_zero_is_preserved():
    x = torch.zeros(20)
    q = make_quant(bits=8)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert torch.all(qx == 0)
    assert torch.all(dx == 0.0)


def test_quantized_values_within_bounds():
    x = torch.linspace(-15.0, 15.0, steps=250)
    q = make_quant(bits=8)

    qx = q.quantize(x)
    assert torch.all(qx >= q.qmin)
    assert torch.all(qx <= q.qmax)


def test_sign_preservation():
    x = torch.tensor([-4.2, -0.05, 0.05, 3.8])
    q = make_quant(bits=8)

    qx = q.quantize(x)
    assert torch.all(torch.sign(qx) == torch.sign(x))


def test_monotonicity():
    x = torch.linspace(-5.0, 5.0, steps=100)
    q = make_quant(bits=8)

    qx = q.quantize(x)
    # Monotonically non-decreasing
    assert torch.all(qx[1:] >= qx[:-1])


def test_small_amplitude_resolution_advantage():
    """
    Core companding thesis: mu-law companding allocates substantially higher
    bit resolution (smaller effective quantization steps) to low-amplitude
    signals near zero than linear uniform quantization of identical bit-width.
    """
    # Low amplitude signals within dynamic range [-1, 1]
    low_amp_x = torch.tensor([0.005, 0.01, 0.02, 0.03, -0.01, -0.025])
    # Full dynamic range reference including peaks
    full_signal = torch.cat([low_amp_x, torch.tensor([1.0, -1.0])])

    # 1. Mu-law quantizer (8 bits)
    q_mulaw = make_quant(bits=8, mu=255.0)
    qx_mulaw = q_mulaw.quantize(full_signal)
    dx_mulaw = q_mulaw.dequantize(qx_mulaw)
    mulaw_errors = (full_signal[: len(low_amp_x)] - dx_mulaw[: len(low_amp_x)]).abs()

    # 2. Linear uniform quantizer (8 bits)
    obs_lin = MinMaxObserver()
    rnd_lin = NearestRounding()
    q_linear = SymmetricUniformQuantizer(bits=8, observer=obs_lin, rounding=rnd_lin)
    qx_lin = q_linear.quantize(full_signal)
    dx_lin = q_linear.dequantize(qx_lin)
    linear_errors = (full_signal[: len(low_amp_x)] - dx_lin[: len(low_amp_x)]).abs()

    # Mu-law should achieve substantially lower quantization error on small signals
    assert mulaw_errors.mean() < linear_errors.mean()


def test_peak_and_scale_reconstruction():
    x = torch.tensor([-10.0, -2.5, 0.0, 2.5, 10.0])
    q = make_quant(bits=8)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert q.scale is not None
    # Endpoints should reconstruct with minimal relative error
    assert torch.isclose(dx[0], x[0], atol=0.2)
    assert torch.isclose(dx[-1], x[-1], atol=0.2)


def test_multidimensional_tensor_shapes():
    # 3D tensor (e.g. batch x channels x audio_samples)
    x = torch.randn(4, 2, 64)
    q = make_quant(bits=8)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert qx.shape == x.shape
    assert dx.shape == x.shape


def test_invalid_parameters():
    obs = MinMaxObserver()
    rnd = NearestRounding()

    with pytest.raises(ValueError, match="mu must be strictly positive"):
        MuLawQuantizer(bits=8, observer=obs, rounding=rnd, mu=-1.0)

    with pytest.raises(ValueError, match="mu must be strictly positive"):
        MuLawQuantizer(bits=8, observer=obs, rounding=rnd, mu=0.0)

    with pytest.raises(ValueError, match="Bits must be at least 2"):
        MuLawQuantizer(bits=1, observer=obs, rounding=rnd, mu=255.0)


def test_dequantize_uninitialized_error():
    q = make_quant(bits=8)
    dummy_qx = torch.tensor([0, 10, -10])
    with pytest.raises(RuntimeError, match="Cannot dequantize before quantize"):
        q.dequantize(dummy_qx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution():
    x = torch.tensor([1.2, -0.7, 0.0, 2.5, -2.5], device="cuda")
    q = make_quant(bits=8)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert qx.is_cuda
    assert dx.is_cuda
    assert torch.all(qx >= q.qmin)
    assert torch.all(qx <= q.qmax)
