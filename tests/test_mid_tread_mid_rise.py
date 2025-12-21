import torch

from inwhale.core.uniform import(
    MidTreadUniformQuantizer,
    MidRiseUniformQuantizer,
)

from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding

def test_mid_tread_has_zero_level():
    observer = MinMaxObserver()
    rounding = NearestRounding()
    quantizer = MidTreadUniformQuantizer(bits=8, observer=observer, rounding=rounding)

    x = torch.tensor([-0.2, 0.0, 0.3])
    qx = quantizer.quantize(x)

    assert torch.any(qx == 0)

def test_mid_rise_has_no_zero_level():
    observer = MinMaxObserver()
    rounding = NearestRounding()
    quantizer = MidRiseUniformQuantizer(bits=8, observer=observer, rounding=rounding)

    x = torch.tensor([-0.2, 0.0, 0.3])
    qx = quantizer.quantize(x)

    assert not torch.any(qx == 0)

    