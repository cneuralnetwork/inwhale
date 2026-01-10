import torch
from .quantizer import BaseQuantizer
from inwhale.observers.minmax import MinMaxObserver


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

    def _compute_scale(self):
        min_val, max_val = self.observer.get_range()

        max_abs = torch.max(min_val.abs(), max_val.abs())

        self.scale = max_abs / self.qmax
        self.scale = torch.clamp(self.scale, min=1e-8)

    def quantize(self, x):
        self.observer.observe(x)

        if self.bits == 1:
            min_val, max_val = self.observer.get_range()
            max_abs = torch.max(min_val.abs(), max_val.abs())
            self.scale = torch.clamp(max_abs, min=1e-8)
            qx = torch.sign(x)
            qx[qx == 0] = 1
            return qx

        self._compute_scale()

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


class DeadZoneSymmetricQuantizer(BaseQuantizer):
    def __init__(self, bits, observer, rounding, threshold_ratio=0.5):
        super().__init__(bits)

        self.observer = observer
        self.rounding = rounding
        self.scale = None
        self.threshold_ratio = threshold_ratio
        self.threshold = None

        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1)) - 1

    def _compute_params(self):
        min_val, max_val = self.observer.get_range()

        if max_val == min_val:
            if isinstance(max_val, torch.Tensor):
                self.scale = torch.ones_like(max_val)
            else:
                self.scale = torch.tensor(1.0)
            self.threshold = self.threshold_ratio * self.scale
            return

        max_abs = torch.max(min_val.abs(), max_val.abs())

        self.scale = max_abs / self.qmax
        self.scale = torch.clamp(self.scale, min=1e-8)

        self.threshold = self.threshold_ratio * self.scale

    def quantize(self, x):
        self.observer.observe(x)

        if self.bits == 1:
            min_val, max_val = self.observer.get_range()
            max_abs = torch.max(min_val.abs(), max_val.abs())
            self.scale = torch.clamp(max_abs, min=1e-8)
            self.threshold = self.threshold_ratio * self.scale

            qx = torch.sign(x)
            qx[qx.abs() < self.threshold] = 0
            return qx

        self._compute_params()

        mask = x.abs() < self.threshold

        sign = torch.sign(x)
        abs_x = x.abs()

        # we shift by threshold, then quantize
        qx = sign * self.rounding.round((abs_x - self.threshold) / self.scale)

        qx = torch.where(mask, torch.zeros_like(qx), qx)
        qx = torch.clamp(qx, self.qmin, self.qmax)

        return qx

    def dequantize(self, qx):
        sign = torch.sign(qx)
        abs_qx = qx.abs()

        dx = sign * (
            abs_qx * self.scale
            + torch.where(qx != 0, self.threshold, torch.zeros_like(self.threshold))
        )

        return dx



class MidTreadUniformQuantizer(BaseQuantizer):
    """
    mid-tread uniform quantizer where zero is a quantization level
    the quantizer typically has an odd number of quantization levels with zero at the central level.
    zero is at a "tread" (flat part) of the staircase.
    this is particularly useful for signals that frequently cross zero.
    formula:
    q = round(x / scale)
    x' = q * scale
    attributes:
    bits : number of bits for quantization
    min : minimum value of input range
    max : maximum value of input range
    scale : quantization step size
    """
    def __init__(self, bits, observer, rounding): # initialize mid-tread quantizer
        super().__init__(bits)
        self.observer = observer
        self.rounding = rounding
        self.scale = None

        self.qmin = -(1 << (bits-1))
        self.qmax = (1 << (bits - 1)) - 1

    def _compute_scale(self):
        min_val, max_val = self.observer.get_range()
        max_abs = torch.max(min_val.abs(), max_val.abs())
        self.scale = torch.clamp(max_abs / self.qmax, min=1e-8)
    
    def quantize(self, x):
        """
        quantize input values using mid_tread quantization.
        x : input values to quantize
        returns quantized value (qx)
        """
        self.observer.observe(x)
        self._compute_scale()

        qx = x / self.scale
        qx = self.rounding.round(qx)
        qx = torch.clamp(qx, self.qmin, self.qmax)
        return qx
    
    def dequantize(self, qx):
        return qx * self.scale
    


class MidRiseUniformQuantizer(BaseQuantizer):
    """
    mid-rise uniform quantizer where zero lies between two quantization levels,
    The quantizer has an even number of quantization levels
    symmetric around the origin.
     Zero is NOT one of the output levels - it sits at a "riser"
    between two quantization levels.
    formula:
    q = floor(x/scale) + 0.5
    x' = q*scale
    """

    def __init__(self, bits, observer, rounding):
        super().__init__(bits)
        self.observer = observer
        self.rounding = rounding
        self.scale = None

        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1))

    def _compute_scale(self):
        min_val, max_val = self.observer.get_range()
        max_abs = torch.max(min_val.abs(), max_val.abs())
        self.scale = torch.clamp(max_abs / self.qmax, min=1e-8)

    def quantize(self, x):
        self.observer.observe(x)
        self._compute_scale()

        qx = x / self.scale - 0.5
        qx = torch.floor(x / self.scale) + 0.5
        qx = torch.clamp(qx, self.qmin, self.qmax)
        return qx
    
    def dequantize(self, qx):
        return qx * self.scale




class PerChannelAsymmetricUniformQuantizer(BaseQuantizer):
    """
    Per-channel quantization applies uniform quantization independently to each channel of the tensor. 
    Each channel has its own scale and zero-point, rather than sharing a single scale and zero-point across the entire tensor.

    why do we need per-channel quantizer, when already per-tensor exists?
    Let a weight tensor has 2 output channels, 
    channel 0: [-1, 1]
    channel 1: [- 10, 15]

    per-tensor quantization uses :  Xmin = -10 and Xmax = 15, so both channels shares he same scale and zero point.
    But in result , channel 1 is OK , but channel 0 gives almost 0 for all the values , because small numbers are crushed by scale chosen for big numbers.
    leading to high precision loss. 
    To fix this , per-channel treats each output channel in a tensor independently.
    For each channel seperate xmin , xmax, scale , zero point is calcuated .
    As a result , per-channel quantization reduces quantization error, specially for tensors where different channels have very different values.
    formulas we will use (the same as per-tensor , but they are applied independently to each channel, than a whole tensor):
    for a tensor X , channels are indexed by c.
    x_min = min(X[c])
    x_max = max(X[c])

    scale[c] = (x_max[c] - x_mi[c]) / (qmax - qmin)
    zero_point[c] = round(qmin - x_min[c] / scale[c]) , then clamped to the rance [qmin, qmax]
    qx = clamp(round(x / scale[c] + zero_point[c]), qmin, qmax)  // quantized value of x
    x' = (q - zero_point[c]) * scale[c] // dequantize
    """
    def __init__(self, bits, observer, rounding, axis=0, signed=False):
        super().__init__(bits)

        self.bits = bits
        self.axis = axis
        self.rounding = rounding
        self.signed = signed

        if signed:
            self.qmin = -(1 << (bits - 1))
            self.qmax = (1 << (bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << bits) - 1

        self.scale = None
        self.zero_point = None


    
    def quantize(self, x):
        """
        Quantize input tensor using per-channel asymmetric uniform quantization.
        """
        # Move channel axis to front for easy iteration
        x_perm = x.movedim(self.axis, 0)
        num_channels = x_perm.shape[0]

        scales = []
        zero_points = []
        qxs = []

        for c in range(num_channels):
            x_c = x_perm[c]

            observer = MinMaxObserver()
            observer.observe(x_c)
            min_val, max_val = observer.get_range()

            if max_val == min_val:
                scale = torch.tensor(1.0, device=x.device)
                zero_point = torch.tensor(self.qmin, device=x.device)
            else:
                scale = (max_val - min_val) / (self.qmax - self.qmin)
                scale = torch.clamp(scale, min=1e-8)

                zp_real = self.qmin - min_val / scale
                zero_point = torch.clamp(
                    self.rounding.round(zp_real), self.qmin, self.qmax
                )

            qx_c = x_c / scale + zero_point
            qx_c = self.rounding.round(qx_c)
            qx_c = torch.clamp(qx_c, self.qmin, self.qmax)

            scales.append(scale)
            zero_points.append(zero_point)
            qxs.append(qx_c)

        # stack per-channel params
        self.scale = torch.stack(scales)
        self.zero_point = torch.stack(zero_points)

        # restore original axis order
        qx = torch.stack(qxs).movedim(0, self.axis)
        return qx

    
    def dequantize(self, qx):
        """
        Dequantize an integer tensor back to floating-point values
        """
        scale = self.scale.view(
            *([1] * self.axis), -1, *([1] * (qx.dim() - self.axis - 1))
        )
        zero_point = self.zero_point.view(
            *([1] * self.axis), -1, *([1] * (qx.dim() - self.axis - 1))
        )
        return (qx - zero_point) * scale

    