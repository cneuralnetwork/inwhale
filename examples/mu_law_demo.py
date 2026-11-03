import warnings
import torch

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch._subclasses.functional_tensor"
)

from inwhale.core.non_uniform import MuLawQuantizer
from inwhale.core.uniform import SymmetricUniformQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding

# Input tensor with high dynamic range (mix of tiny signals and large peaks)
x = torch.tensor([0.005, -0.012, 0.045, -0.150, 0.850, -1.000])

observer = MinMaxObserver()
rounding = NearestRounding()

# Standard 8-bit mu-law quantizer (mu=255, standardized in ITU-T G.711 & WaveNet)
mu_quant = MuLawQuantizer(
    bits=8,
    observer=observer,
    rounding=rounding,
    mu=255.0,
)

"""
Mu-law Companding (Compressing + Expanding) Intuition:

Continuous signals like speech, audio, and neural activations typically cluster heavily
near zero with rare extreme peaks. Uniform linear quantization allocates an equal step
size across all ranges, wasting precision on rare peaks while causing high relative
error on delicate low-amplitude features.

Mu-law companding solves this:
1. Normalize input x to [-1, 1] using observed dynamic range.
2. Non-linear logarithmic compression:
       F(x) = sign(x) * ln(1 + mu * |x|) / ln(1 + mu)
   Near zero, the slope is mu / ln(1 + mu) ~ 46, effectively zooming in on small values.
3. Uniform quantization in the compressed domain into [qmin, qmax].
4. Dequantization and inverse exponential expansion:
       F^-1(y) = sign(y) * (1 / mu) * ((1 + mu)^|y| - 1)
   restoring full dynamic range without clipping.
"""

# Quantize and dequantize with Mu-law
qx_mu = mu_quant.quantize(x)
dx_mu = mu_quant.dequantize(qx_mu)
error_mu = (x - dx_mu).abs()

# Compare against linear uniform quantization at the exact same 8-bit budget
lin_obs = MinMaxObserver()
lin_rnd = NearestRounding()
lin_quant = SymmetricUniformQuantizer(bits=8, observer=lin_obs, rounding=lin_rnd)
qx_lin = lin_quant.quantize(x)
dx_lin = lin_quant.dequantize(qx_lin)
error_lin = (x - dx_lin).abs()

print("=" * 65)
print("INWHALE: MU-LAW COMPANDING DEMO (8-bit, mu=255.0)")
print("=" * 65)
print(f"Original Input:      {x.tolist()}")
print(f"Quantized (Integer): {qx_mu.tolist()}")
print(f"Dequantized (Float): {[round(v, 5) for v in dx_mu.tolist()]}")
print(f"Mu-Law Absolute Err: {[round(v, 5) for v in error_mu.tolist()]}")
print(f"Linear Absolute Err: {[round(v, 5) for v in error_lin.tolist()]}")
print("-" * 65)
print(
    f"Mean Absolute Error on small signals (|x| <= 0.05):\n"
    f"  -> Mu-Law error: {error_mu[:3].mean():.6f}\n"
    f"  -> Linear error: {error_lin[:3].mean():.6f} (Mu-law has {(error_lin[:3].mean() / error_mu[:3].mean()):.1f}x lower error!)"
)
print("=" * 65)
