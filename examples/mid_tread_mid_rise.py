import torch

from inwhale.core.uniform import (
    MidTreadUniformQuantizer,
    MidRiseUniformQuantizer,
)
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding

def main():
    x  = torch.tensor([-1.2, -0.3, 0.0, 0.4, 1.7])

    observer = MinMaxObserver()
    rounding = NearestRounding()

    mid_tread = MidTreadUniformQuantizer(bits=8, observer=observer, rounding=rounding)
    qx_tread = mid_tread.quantize(x)
    dx_tread = mid_tread.dequantize(qx_tread)

    print("Mid-tread quantization: ")
    print("x =", x)
    print("qx = ", qx_tread)
    print("x' = ", dx_tread)
    print()

    observer = MinMaxObserver()
    mid_rise = MidRiseUniformQuantizer(bits=8, observer=observer, rounding=rounding)
    qx_rise = mid_rise.quantize(x)
    dx_rise = mid_rise.dequantize(qx_rise)

    print("Mid-rise quantization:")
    print("x = ", x)
    print("qx = ", qx_rise)
    print("x' =", dx_rise)

if __name__ == "__main__":
    main()

        