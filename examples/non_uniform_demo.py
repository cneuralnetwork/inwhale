import warnings
import torch

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch._subclasses.functional_tensor"
)

from inwhale.core.non_uniform import LogarithmicQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


# Input tensor
x = torch.tensor([1.013, 2.513, -3.264, 0.235, -0.251, 0.012])

observer = MinMaxObserver()
rounding = NearestRounding()

quant = LogarithmicQuantizer(
    bits=8,
    observer=observer,
    rounding=rounding,
)

"""
Logarithmic quantization intuition:

We represent values as:
    sign(x) * 2^k

Steps:
1. Observer records min/max of x
2. Compute max_abs = max(|min|, |max|)
3. Convert max_abs to log2 domain
4. Round exponent and clamp to representable exponent range
5. For each non-zero x:
   - k = round(log2(|x|))
   - qx = sign(x) * 2^k

Zero stays zero.

This favors scale invariance over linear precision.
"""

qx = quant.quantize(x)
dx = quant.dequantize(qx)

print("Original:", x)
print("Quantized:", qx)
print("Dequantized:", dx)
print("Absolute error:", (x - dx).abs())
