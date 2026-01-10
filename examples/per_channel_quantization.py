import torch

from inwhale.core.uniform import PerChannelAsymmetricUniformQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding

x = torch.tensor([
    [-1.0, 1.0, 0.5, -0.5],
    [-10.0, 15.0, 8.0, -5,0],
])
observer = MinMaxObserver(axis=0)
rounding = NearestRounding()

q = PerChannelAsymmetricUniformQuantizer(
    bits=8, observer=observer, rounidng=rounding, axis=0, signed=True,
)

"""
what the oberver sees (perchannel)
channel 0:
min = -1.0, max = 1.0
channel 1:
min = -10.0, max = 15.0

for 8-bit signed quantization:
qmin = -128, qmax = 127

scale computation:
channel 0:
scale[0] = min - max / qmax - qmin = 2 / 255 = 0.00784
channel 1:
scale[1] = 15 - (-10) / 255 = 0.09804

zero_point[c] = round(qmin - min / scale)

//quantization
q = round(x / scale) + zero_point

//dequantization
x' = (q - zero_point) * scale

"""
qx = q.quantize(x)
dx = q.dequantize(qx)

print("Original:")
print(x)

print("Quantized (per-channel):")
print(qx)

print("Dequantized:")
print(dx)

print("Absolute error:")
print((x - dx).abs())

print("Per-channel scales:")
print(q.scale)

print("Per-channel zero-points:")
print(q.zero_point)