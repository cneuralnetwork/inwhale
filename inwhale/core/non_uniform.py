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


class ALawQuantizer(BaseQuantizer):
    """
    A-law Companding Non-Uniform Quantizer.

    Companding (Compressing + Expanding) is a classic non-uniform quantization
    scheme standardized in ITU-T Recommendation G.711. It is predominantly used
    in European and international digital telecommunications (PCM 30 standard),
    and widely applied in neural audio compression, deep speech synthesis, and
    non-uniform weight/gradient quantization in deep learning.

    Why non-uniform / companding?
    In many empirical signals and neural network representations (speech audio,
    activation tensors, and weight gradients), value distributions cluster heavily
    around zero, exhibiting heavy tails with rare large-amplitude peaks.
    Furthermore, human perceptual mechanisms adhere to the Weber-Fechner law:
    perceptual sensitivity is logarithmic, meaning relative distortion at low
    amplitudes is far more severe than distortion at peak amplitudes.

    Linear (uniform) quantization imposes a uniform step size throughout the entire
    dynamic range, dedicating the majority of quantization bins to rarely visited
    extremes while introducing large relative error near zero.

    Companding resolves this via a 3-stage non-linear pipeline:
        Signal x
          -> Dynamic range normalization to [-1, 1] via peak scale gamma
          -> Piecewise A-law logarithmic compression F_A(x)
          -> Uniform integer quantization in compressed domain
          -> Dequantization to normalized compressed domain
          -> Inverse A-law expansion F_A^-1(y)
          -> Rescaling to original dynamic range.

    Mathematical Formulation:
    For normalized input x in [-1, 1] and companding parameter A >= 1.0 (ITU-T G.711
    standard value is A = 87.6):

        When |x| < 1 / A:
            F_A(x) = sgn(x) * (A * |x|) / (1 + ln(A))

        When 1 / A <= |x| <= 1:
            F_A(x) = sgn(x) * (1 + ln(A * |x|)) / (1 + ln(A))

    Key Mathematical Properties:
    - Exact Zero Preservation: F_A(0) = 0.
    - Exact Boundary Alignment: F_A(1) = 1, F_A(-1) = -1.
    - Continuous Transition: At |x| = 1 / A,
        Linear branch: (A * (1 / A)) / (1 + ln(A)) = 1 / (1 + ln(A))
        Logarithmic branch: (1 + ln(A * (1 / A))) / (1 + ln(A)) = 1 / (1 + ln(A))
        Both branches match seamlessly with no discontinuity.
    - Small Signal Amplification: For |x| < 1 / A, the slope is constant:
        dF_A / dx = A / (1 + ln(A)) approx 87.6 / (1 + 4.4728) approx 16.0
        yielding a ~16x resolution enhancement (or ~24 dB dynamic range improvement)
        for small amplitudes near zero compared to linear quantization.
    - Identity Limit: When A = 1.0, ln(1) = 0 and 1 / A = 1, reducing F_A(x) to
      the linear identity transformation F(x) = x.

    Inverse Expansion Function:
    With compressed domain threshold y_th = 1 / (1 + ln(A)):

        When |y| < y_th:
            F_A^-1(y) = sgn(y) * (|y| * (1 + ln(A))) / A

        When y_th <= |y| <= 1:
            F_A^-1(y) = sgn(y) * exp(|y| * (1 + ln(A)) - 1) / A

    Quantization Levels:
    Values in compressed domain [-1, 1] are mapped to signed integer grid [qmin, qmax]
    where qmin = -(1 << (bits - 1)) and qmax = (1 << (bits - 1)) - 1.
    """

    def __init__(self, bits: int, observer, rounding, A: float = 87.6):
        super().__init__(bits)
        if A < 1.0:
            raise ValueError("A must be at least 1.0 (typically 87.6).")
        if bits < 2:
            raise ValueError("Bits must be at least 2 for signed A-law quantization.")

        self.observer = observer
        self.rounding = rounding
        self.A = float(A)
        self.scale = None

        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1)) - 1

    def _compute_scale(self):
        min_val, max_val = self.observer.get_range()
        max_abs = torch.max(min_val.abs(), max_val.abs())
        self.scale = torch.clamp(max_abs, min=1e-8)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantizes input tensor x using A-law companding.

        Args:
           x: Input floating-point tensor of arbitrary shape.

        Returns:
           qx: Quantized tensor with integer values in [qmin, qmax].
        """
        self.observer.observe(x)
        self._compute_scale()

        # Step 1: Normalize signal to [-1.0, 1.0]
        x_norm = torch.clamp(x / self.scale, -1.0, 1.0)
        abs_x = x_norm.abs()
        sign_x = torch.sign(x_norm)

        # Step 2: Apply piecewise A-law compression
        A_t = torch.tensor(self.A, dtype=x.dtype, device=x.device)
        one_plus_ln_A = 1.0 + torch.log(A_t)
        inv_A = 1.0 / A_t

        linear_part = (A_t * abs_x) / one_plus_ln_A
        # Clamp argument inside log to 1.0 to avoid log(0) or domain errors in unselected branches
        log_part = (1.0 + torch.log(torch.clamp(A_t * abs_x, min=1.0))) / one_plus_ln_A

        y_mag = torch.where(abs_x < inv_A, linear_part, log_part)
        y = sign_x * y_mag

        # Step 3: Uniformly quantize in compressed domain [-1.0, 1.0] to [qmin, qmax]
        qy = y * self.qmax
        qx = self.rounding.round(qy)
        qx = torch.clamp(qx, self.qmin, self.qmax)

        return qx

    def dequantize(self, qx: torch.Tensor) -> torch.Tensor:
        """
        Dequantizes tensor qx using inverse A-law expansion.

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
        abs_y = y_hat.abs()
        sign_y = torch.sign(y_hat)

        # Step 2: Apply inverse A-law expansion
        A_t = torch.tensor(self.A, dtype=y_hat.dtype, device=y_hat.device)
        one_plus_ln_A = 1.0 + torch.log(A_t)
        y_th = 1.0 / one_plus_ln_A

        linear_part = (abs_y * one_plus_ln_A) / A_t
        log_part = torch.exp(abs_y * one_plus_ln_A - 1.0) / A_t

        x_norm_mag = torch.where(abs_y < y_th, linear_part, log_part)
        x_norm_hat = sign_y * x_norm_mag

        # Step 3: Rescale back to original dynamic range
        return x_norm_hat * self.scale
