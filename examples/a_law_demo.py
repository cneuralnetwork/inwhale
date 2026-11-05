import warnings
import torch

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch._subclasses.functional_tensor"
)

from inwhale.core.non_uniform import ALawQuantizer
from inwhale.core.uniform import SymmetricUniformQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


# Signal with both low-amplitude details (clustering near zero) and high-amplitude peaks
x = torch.tensor([0.002, -0.008, 0.025, -0.090, 0.450, -1.000])

# Configure 8-bit A-law Quantizer (ITU-T G.711, A = 87.6)
obs_alaw = MinMaxObserver()
rnd_alaw = NearestRounding()
quant_alaw = ALawQuantizer(bits=8, observer=obs_alaw, rounding=rnd_alaw, A=87.6)

# Configure 8-bit Symmetric Linear Uniform Quantizer for comparison
obs_lin = MinMaxObserver()
rnd_lin = NearestRounding()
quant_lin = SymmetricUniformQuantizer(bits=8, observer=obs_lin, rounding=rnd_lin)

"""
A-law Companding Intuition:
Standardized in ITU-T G.711 for digital telecommunications (PCM 30), A-law
companding is a piecewise non-uniform quantization scheme designed for signals
concentrated around zero (speech audio, neural activations, gradients).

Piecewise mapping for normalized input x in [-1, 1] with parameter A = 87.6:
1. Linear region (|x| < 1/A approx 0.0114):
   F_A(x) = sgn(x) * (A * |x|) / (1 + ln(A))
   Slope = A / (1 + ln(A)) approx 16.0 (amplifies fine signals ~16x)

2. Logarithmic region (1/A <= |x| <= 1):
   F_A(x) = sgn(x) * (1 + ln(A * |x|)) / (1 + ln(A))

This allocates dense quantization levels to small magnitudes while compressing
large peaks, matching human perceptual logarithmic sensitivity (Weber-Fechner law).
"""

# Quantize and dequantize with A-law
qx_alaw = quant_alaw.quantize(x)
dx_alaw = quant_alaw.dequantize(qx_alaw)
err_alaw = (x - dx_alaw).abs()

# Quantize and dequantize with Linear Uniform
qx_lin = quant_lin.quantize(x)
dx_lin = quant_lin.dequantize(qx_lin)
err_lin = (x - dx_lin).abs()

print("=" * 76)
print("A-law Companding (A = 87.6, 8-bit) vs. Linear Uniform Quantization (8-bit)")
print("=" * 76)
print(
    f"{'Original (x)':<14}{'A-law Code':<12}{'A-law Reconstructed':<22}{'A-law Error':<14}{'Linear Error':<14}"
)
print("-" * 76)

for orig, q_a, d_a, e_a, e_l in zip(x, qx_alaw, dx_alaw, err_alaw, err_lin):
    print(
        f"{orig.item():<14.5f}{int(q_a.item()):<12}{d_a.item():<22.5f}{e_a.item():<14.6f}{e_l.item():<14.6f}"
    )

print("-" * 76)
low_amp_mask = x.abs() <= 0.03
mean_err_alaw = err_alaw[low_amp_mask].mean().item()
mean_err_lin = err_lin[low_amp_mask].mean().item()
ratio = mean_err_lin / max(mean_err_alaw, 1e-12)

print(f"Mean Abs Error on small signals (|x| <= 0.03):")
print(f"  A-law Quantizer:  {mean_err_alaw:.6f}")
print(f"  Linear Quantizer: {mean_err_lin:.6f}")
print(f"  Resolution boost: {ratio:.1f}x lower distortion near zero with A-law")
print("=" * 76)
