import torch

from inwhale.core.uniform import AsymmetricUniformQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


def test_unsigned_qmin_qmax_bits8():
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = AsymmetricUniformQuantizer(bits=8, observer=obs, rounding=rnd, signed=False)
    assert q.qmin == 0
    assert q.qmax == 255


def test_signed_qmin_qmax_bits8():
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = AsymmetricUniformQuantizer(bits=8, observer=obs, rounding=rnd, signed=True)
    assert q.qmin == -128
    assert q.qmax == 127


def test_zero_range_edge_case_sets_scale_and_zero_point():
    x = torch.full((10,), 3.14)
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = AsymmetricUniformQuantizer(bits=8, observer=obs, rounding=rnd, signed=False)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    # when min==max, implementation sets scale=1.0 and zero_point=qmin
    assert float(q.scale) == 1.0
    assert q.zero_point == q.qmin
    # with scale=1 and zero_point=qmin, dequantized equals rounded x
    assert torch.allclose(dx, torch.round(x), atol=1e-6)


def test_params_computation_and_round_trip_accuracy():
    x = torch.tensor([-1.0, 0.0, 2.0, 3.5])
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = AsymmetricUniformQuantizer(bits=8, observer=obs, rounding=rnd, signed=False)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert q.scale is not None
    assert q.zero_point is not None
    # dx close to x within one scale step
    assert torch.allclose(dx, x, atol=float(q.scale))


def test_quantized_values_within_range_unsigned():
    x = torch.linspace(-10, 10, steps=101)
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = AsymmetricUniformQuantizer(bits=8, observer=obs, rounding=rnd, signed=False)

    qx = q.quantize(x)
    assert torch.all(qx >= q.qmin)
    assert torch.all(qx <= q.qmax)
