import torch

from inwhale.core.non_uniform import LogarithmicQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


def make_quant(bits=4):
    obs = MinMaxObserver()
    rnd = NearestRounding()
    return LogarithmicQuantizer(bits=bits, observer=obs, rounding=rnd)


def test_exponent_boundaries():
    q = make_quant(bits=4)

    assert q.emin == -(1 << (4 - 1))
    assert q.emax == (1 << (4 - 1)) - 1


def test_zero_is_preserved():
    x = torch.zeros(10)
    q = make_quant(bits=4)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert torch.all(qx == 0)
    assert torch.all(dx == 0)


def test_quantized_values_are_powers_of_two():
    x = torch.tensor([0.25, 0.5, 1.0, 2.0, 3.0, 6.0])
    q = make_quant(bits=5)

    qx = q.quantize(x)

    non_zero = qx[qx != 0].abs()
    log2_vals = torch.log2(non_zero)

    # Quantized magnitudes must be integer powers of two
    assert torch.allclose(log2_vals, torch.round(log2_vals))


def test_sign_is_preserved():
    x = torch.tensor([-0.5, -2.0, 1.5, 4.0])
    q = make_quant(bits=4)

    qx = q.quantize(x)

    assert torch.all(torch.sign(qx) == torch.sign(x))


def test_larger_magnitude_monotonicity():
    x = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])
    q = make_quant(bits=4)

    qx = q.quantize(x)

    # Log quantization should preserve ordering by magnitude
    assert torch.all(qx[:-1].abs() <= qx[1:].abs())


def test_observer_is_used():
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = LogarithmicQuantizer(bits=4, observer=obs, rounding=rnd)

    x1 = torch.tensor([1.0, 2.0])
    q.quantize(x1)

    min1, max1 = obs.get_range()

    x2 = torch.tensor([0.5, 8.0])
    q.quantize(x2)

    min2, max2 = obs.get_range()

    assert min2 <= min1
    assert max2 >= max1
