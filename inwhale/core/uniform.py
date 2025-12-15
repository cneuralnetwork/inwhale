import torch
from .quantizer import BaseQuantizer


class SymmetricUniformQuantizer(BaseQuantizer):
    """
    Docstring for SymmetricUniformQuantizer

    The fundamental equation behind symmetric uniform quantization is:
    q = clamp(round(x / scale), qmin, qmax)
    and x' = q * scale
    where scale is computed based on the observed min and max values of the input tensor.

    In real networks, we quantize activations layer by layer and sometimes over many batches, before seeing all inputs.
    So we need a component whose ONLY job is to watch tensors go by and summarize their numeric range. This is the observer.

    And the MinMaxObserver answers one simple questions: What is the smallest and largest value I have ever seen?

    min_val, max_val = self.observer.get_range(), now we can compute the scale.

    Now calculating the scale is quite straightforward.
    Since we wish to _scale_ max_abs_value to qmax, we can just:

    scale = max_abs / qmax
    equivalently,
    max_abs = torch.max(min_val.abs(), max_val.abs())

    Finally, we need to ensure that scale is not zero (which would lead to division by zero during quantization).

    For which we did
    torch.clamp(self.scale, min=1e-8)

    Now one question still remains, why is rounding abstracted?
    Because, rounding is NOT JUST round(), it can change bias and the error distribution.

    There can be multiple strategies for rounding, and they can affect accumulated wrror, training stability, bias towards zero, etc.
    """

    def __init__(self, bits, observer, rounding):
        super().__init__(bits)

        self.observer = observer
        self.rounding = rounding
        self.scale = None

        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1)) - 1

    def _compute_scale(self, x):
        min_val, max_val = self.observer.get_range()

        max_abs = torch.max(min_val.abs(), max_val.abs())

        self.scale = max_abs / self.qmax
        self.scale = torch.clamp(self.scale, min=1e-8)

    def quantize(self, x):
        self.observer.observe(x)
        self._compute_scale(x)

        qx = x / self.scale
        qx = self.rounding.round(qx)
        qx = torch.clamp(qx, self.qmin, self.qmax)
        return qx

    def dequantize(self, qx):
        return qx * self.scale


class AsymmetricUniformQuantizer(BaseQuantizer):
    """
    Docstring for AsymmetricUniformQuantizer

    While symmetric quantization assumes the data is centered around zero, that doesnt happen all the time.
    For example, for ReLU activations, we get something like x belongs to [0, +something].
    Bias terms: x belongs to [3.2, 7.9]

    If we try to force symmetry, we get [-7.9, +7.9]. WE WASTE PRECISION.
    Half of our integer range represents values that never occur. So we abandon symmetry and allow TRANSLATION via zero_point.

    for symmetric mapping we did x = q x scale
    and for asymmtetric we do x = (q - zero_point(THIS, is the OFFSET)) x scale


    (real range length) = (integer range length) x scale
    scale = (max_val - min_val) / (qmax - qmin)
    this means every integer increment changes the real value by 'scale', the entire float range is covered.

    Now we need to think of alignment, we need to slide the integer ruler left or right so that smallest real value must map to the smallest integer value.
    Essentially, min_val => qmin

    any uniform quantizer is a straight line, q = (x / scale) + offset
    we divide by spacing and shift to align.

    AND NOW we enforce the left end alignment:
    we substitute min_val for x and qmin for q:
    qmin = (min_val / scale) + offset
    offset = qmin - (min_val / scale)
    offset is what we call zero_point, rounded to nearest integer within [qmin, qmax]

    EXAMPLE:
    min_val = 2
    max_val = 6
    bits = 8, qmin = 0, qmax = 255
    scale = (6 - 2) / 255 = 4 / 255 = 0.0157 (approx)

    Where would 2 land without offset?
    q = 2 / 0.0157 = 127.4
    but we want 2 => 0, so we need to shift left by 127.4
    That shift IS the offset.
    offset = 0 - 127.4 = -127.4
    and rounding off, zero_point = -127

    let's verify:
    q = 2 / 0.0157 - 127 = 0 (approx)
    q = 6 / 0.0157 - 127 = 255 (approx)

    from here, we can see that symmetric quantization is just a special case of asymmetric quantization where the ruler is already centered.
    """

    def __init__(self, bits, observer, rounding, signed=False):
        super().__init__(bits)

        if signed:
            self.qmin = -(1 << (bits - 1))
            self.qmax = (1 << (bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << bits) - 1

        self.observer = observer
        self.rounding = rounding
        self.scale = None
        self.zero_point = None

    def _compute_params(self):
        min_val, max_val = self.observer.get_range()

        if max_val == min_val:
            self.scale = 1.0
            self.zero_point = self.qmin
            return

        self.scale = (max_val - min_val) / (self.qmax - self.qmin)
        self.scale = torch.clamp(self.scale, min=1e-8)

        zero_point_real = self.qmin - min_val / self.scale
        self.zero_point = torch.clamp(
            self.rounding.round(zero_point_real), self.qmin, self.qmax
        )

    def quantize(self, x):
        self.observer.observe(x)
        self._compute_params()

        qx = x / self.scale + self.zero_point
        qx = self.rounding.round(qx)
        qx = torch.clamp(qx, self.qmin, self.qmax)
        return qx

    def dequantize(self, qx):
        return (qx - self.zero_point) * self.scale
