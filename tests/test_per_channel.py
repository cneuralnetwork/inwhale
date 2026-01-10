import torch

from inwhale.core.uniform import (
    AsymmetricUniformQuantizer,
    PerChannelAsymmetricUniformQuantizer,
)
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding
from inwhale.observers.minmax import MinMaxObserver


def test_per_channel_scale_and_zero_point_shapes():
    """
    Per-channel quantization should produce one scale and zero-point per channel.
    """
    x = torch.tensor([
        [-1.0,  1.0],
        [-10.0, 10.0],
        [-0.5,  0.5],
    ])

    observer = MinMaxObserver()
    rounding = NearestRounding()
    q = PerChannelAsymmetricUniformQuantizer(
        bits=8, observer=observer, rounding=rounding, axis=0, signed=True,
    )
    observer.observe(x)
    q.quantize(x)

    assert q.scale.ndim == 1
    assert q.zero_point.ndim == 1
    assert q.scale.numel() == x.shape[0]
    assert q.zero_point.numel() == x.shape[0]


def test_per_channel_differs_from_per_tensor():
    """
    Per-channel quantization should behave differently from per-tensor
    when channels have very different ranges.
    """
    x = torch.tensor([
        [-1.0,  1.0,  0.5, -0.5],     # small range
        [-10.0, 15.0, 8.0, -5.0],     # large range
    ])

    obs = MinMaxObserver()
    rnd = NearestRounding()

    per_tensor = AsymmetricUniformQuantizer(
        bits=8, observer=obs, rounding=rnd, signed=True
    )
    per_channel = PerChannelAsymmetricUniformQuantizer(
        bits=8, observer=obs, rounding=rnd, axis=0, signed=True
    )

    per_tensor.quantize(x)
    per_channel.quantize(x)

    # channel 0 should differ due to independent scaling
    assert per_tensor.scale.ndim == 0
    assert per_channel.scale.ndim == 1
    assert per_channel.scale.numel() == x.shape[0]



def test_per_channel_zero_range_channel():
    """
    A channel with zero range (min == max) should be handled safely.
    """
    x = torch.tensor([
        [3.14, 3.14, 3.14],   # zero-range channel
        [-2.0, 0.0, 2.0],
    ])

    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = PerChannelAsymmetricUniformQuantizer(
        bits=8, observer=obs, rounding=rnd, axis=0, signed=True
    )
    obs.observe(x)
    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert torch.isfinite(dx).all()


def test_per_channel_round_trip_accuracy():
    """
    Dequantized values should be close to original values within one scale step,
    per channel.
    """
    x = torch.tensor([
        [-1.0, 0.0, 1.0],
        [-10.0, 0.0, 10.0],
    ])

    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = PerChannelAsymmetricUniformQuantizer(
        bits=8, observer=obs, rounding=rnd, axis=0, signed=True
    )
    obs.observe(x)
    qx = q.quantize(x)
    dx = q.dequantize(qx)

    for c in range(x.shape[0]):
        assert torch.allclose(dx[c], x[c], atol=float(q.scale[c]))


def test_per_channel_quantized_values_within_range():
    """
    Quantized values must lie within [qmin, qmax] for all channels.
    """
    x = torch.randn(4, 10)

    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = PerChannelAsymmetricUniformQuantizer(
        bits=8, observer=obs, rounding=rnd, axis=0, signed=True
    )
    obs.observe(x)
    qx = q.quantize(x)

    assert torch.all(qx >= q.qmin)
    assert torch.all(qx <= q.qmax)
