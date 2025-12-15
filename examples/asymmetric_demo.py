import warnings
import torch

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch._subclasses.functional_tensor"
)

from inwhale.core.uniform import AsymmetricUniformQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


# ReLU like activations
x = torch.tensor([0.1, 0.5, 1.2, 2.3, 3.8, 4.9])

observer = MinMaxObserver()
rounding = NearestRounding()
quant = AsymmetricUniformQuantizer(
    bits=8, observer=observer, rounding=rounding, signed=False
)

"""
Observer sees:
min = 0.1
max = 4.9

For 8-bit unsigned:
qmin = 0
qmax = 255

scale = (max - min) / (qmax - qmin)
      = (4.9 - 0.1) / 255
      ≈ 0.0188

zero_point = qmin - min / scale
           ≈ -5.3 -> rounded -> clamped to 0

So:
0.1 -> 0
4.9 -> 255

The entire integer range is used, unlike symmetric quantization.
"""

qx = quant.quantize(x)
dx = quant.dequantize(qx)

print("Original:", x)
print("Quantized:", qx)
print("Dequantized:", dx)
print("Absolute error:", (x - dx).abs())


# Biased activations (narrow positive range)

x = torch.tensor([3.2, 3.5, 4.1, 5.0, 5.8, 7.9])

observer = MinMaxObserver()
quant = AsymmetricUniformQuantizer(
    bits=8, observer=observer, rounding=rounding, signed=False
)

"""
Observer sees:
min = 3.2
max = 7.9

scale = (7.9 - 3.2) / 255 ≈ 0.0184
zero_point = -3.2 / scale ≈ -174 -> clamped to 0

Again, the full [0, 255] range is used to represent only the
relevant data interval.
"""

qx = quant.quantize(x)
dx = quant.dequantize(qx)

print("\nOriginal:", x)
print("Quantized:", qx)
print("Dequantized:", dx)
print("Absolute error:", (x - dx).abs())


# signed asymmetric (off-centered range)

x = torch.tensor([-2.0, -1.0, 0.5, 1.5, 2.0, 3.5])

observer = MinMaxObserver()
quant = AsymmetricUniformQuantizer(
    bits=8, observer=observer, rounding=rounding, signed=True
)

"""
Observer sees:
min = -2.0
max = 3.5

For 8-bit signed:
qmin = -128
qmax = 127

scale = (3.5 - (-2.0)) / 255 ≈ 0.0216
zero_point = qmin - min / scale
           = -35(approx)

This is asymmetric despite being signed: zero is not centered.
"""

qx = quant.quantize(x)
dx = quant.dequantize(qx)

print("\nOriginal:", x)
print("Quantized:", qx)
print("Dequantized:", dx)
print("Absolute error:", (x - dx).abs())
