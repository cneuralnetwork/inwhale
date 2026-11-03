import torch

from .quantizer import BaseQuantizer


class LogarithmicQuantizer(BaseQuantizer):
    def __init__(self, bits, observer, rounding):
        super().__init__(bits)
        self.observer = observer
        self.rounding = rounding

        self.emin = -(1 << (bits - 1))
        self.emax = (1 << (bits - 1)) - 1

    def _compute_scale(self):
        min_val, max_val = self.observer.get_range()

        max_abs = torch.max(min_val.abs(), max_val.abs())
        max_abs = torch.clamp(max_abs, min=1e-8)

        max_exp = torch.log2(max_abs)
        self.exp_max = torch.clamp(self.rounding.round(max_exp), self.emin, self.emax)

        self.exp_min = self.emin

    def quantize(self, x):
        self.observer.observe(x)
        self._compute_scale()

        qx = torch.zeros_like(x)
        non_zero = x != 0  # (x != 0).float()

        sign_x = torch.sign(x[non_zero])
        abs_x = x[non_zero].abs()

        log_x = torch.log2(abs_x)
        k = self.rounding.round(log_x)
        k = torch.clamp(k, self.exp_min, self.exp_max)

        qx[non_zero] = sign_x * torch.pow(2.0, k)

        return qx

    def dequantize(self, qx):
        return qx


class MuLawQuantizer(BaseQuantizer):
    """
    Docstring for MuLawQuantizer

    Companding (Compressing + Expanding) is a classic non-uniform quantization
    scheme standardized in ITU-T G.711 and heavily utilized in deep learning audio,
    speech synthesis, and waveform generation (such as DeepMind's WaveNet).

    Why non-uniform / companding?
    In many continuous signals (speech, audio waveforms, neural network gradients,
    and activation tensors), values are heavily concentrated around zero, with large
    peaks occurring relatively infrequently. Furthermore, human perceptual systems
    follow the Weber-Fechner law: perceptual sensitivity is logarithmic, meaning
    relative error matters far more at small amplitudes than at high peak levels.

    Linear (uniform) quantization applies an identical step size across all magnitudes,
    wasting bit allocation on rare peak values while incurring large relative
    quantization distortion on fine low-amplitude signals.

    Companding resolves this via a 3-step pipeline:
        Signal x
          -> Dynamic range normalization to [-1, 1]
          -> Logarithmic compression F(x)
          -> Uniform quantization Q(y) in compressed domain
          -> Dequantization Q^-1(q)
          -> Exponential expansion F^-1(y_hat)
          -> Rescaling to original dynamic range.

    Mathematical Formulation:
    For normalized input x in [-1, 1] and companding factor mu > 0 (typically mu = 255.0):
        F(x) = sgn(x) * ln(1 + mu * |x|) / ln(1 + mu)

    Notice:
    - F(0) = 0 (exact zero preservation)
    - F(1) = 1, F(-1) = -1 (boundary alignment)
    - Derivative near zero: F'(0) = mu / ln(1 + mu) approx 46 for mu = 255, giving
      ~46x higher resolution (finer step size) to small values compared to uniform quantization.

    The inverse expansion function F^-1(y) for y in [-1, 1] is:
        F^-1(y) = sgn(y) * (1 / mu) * ((1 + mu)^|y| - 1)

    Quantization levels:
    In the compressed domain [-1, 1], values are mapped onto a signed integer grid
    [qmin, qmax] where qmin = -(1 << (bits - 1)) and qmax = (1 << (bits - 1)) - 1.
    """

    def __init__(self, bits: int, observer, rounding, mu: float = 255.0):
        super().__init__(bits)
        if mu <= 0:
            raise ValueError("mu must be strictly positive (typically 255.0).")
        if bits < 2:
            raise ValueError("Bits must be at least 2 for signed mu-law quantization.")

        self.observer = observer
        self.rounding = rounding
        self.mu = float(mu)
        self.scale = None

        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1)) - 1

    def _compute_scale(self):
        min_val, max_val = self.observer.get_range()
        max_abs = torch.max(min_val.abs(), max_val.abs())
        self.scale = torch.clamp(max_abs, min=1e-8)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantizes input tensor x using mu-law companding.

        Args:
            x: Input floating-point tensor of arbitrary shape.

        Returns:
            qx: Quantized tensor with values in [qmin, qmax].
        """
        self.observer.observe(x)
        self._compute_scale()

        # Step 1: Normalize signal to [-1.0, 1.0]
        x_norm = torch.clamp(x / self.scale, -1.0, 1.0)

        # Step 2: Apply mu-law compression
        mu_t = torch.tensor(self.mu, dtype=x.dtype, device=x.device)
        ln_1_plus_mu = torch.log(1.0 + mu_t)
        y = torch.sign(x_norm) * torch.log(1.0 + self.mu * x_norm.abs()) / ln_1_plus_mu

        # Step 3: Uniformly quantize in compressed domain [-1.0, 1.0] to [qmin, qmax]
        qy = y * self.qmax
        qx = self.rounding.round(qy)
        qx = torch.clamp(qx, self.qmin, self.qmax)

        return qx

    def dequantize(self, qx: torch.Tensor) -> torch.Tensor:
        """
        Dequantizes tensor qx using inverse mu-law expansion.

        Args:
            qx: Quantized tensor with integer values in [qmin, qmax].

        Returns:
            dx: Reconstructed floating-point tensor in original scale.
        """
        if self.scale is None:
            raise RuntimeError(
                "Cannot dequantize before quantize has established scale."
            )

        # Step 1: Map [qmin, qmax] back to normalized compressed domain [-1.0, 1.0]
        y_hat = torch.clamp(qx.float() / self.qmax, -1.0, 1.0)

        # Step 2: Apply inverse mu-law expansion
        mu_t = torch.tensor(self.mu, dtype=y_hat.dtype, device=y_hat.device)
        x_norm_hat = (
            torch.sign(y_hat)
            * (1.0 / self.mu)
            * (torch.pow(1.0 + mu_t, y_hat.abs()) - 1.0)
        )

        # Step 3: Rescale back to original dynamic range
        return x_norm_hat * self.scale
