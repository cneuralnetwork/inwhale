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
