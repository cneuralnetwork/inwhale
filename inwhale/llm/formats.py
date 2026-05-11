import torch
from inwhale.core.quantizer import BaseQuantizer

class NF4Quantizer(BaseQuantizer):
    """
    NormalFloat4 (NF4) Quantizer.
    
    This implements the NF4 format as described in the QLoRA paper (Dettmers et al., 2023).
    NF4 is an information-theoretically optimal data type for normally distributed weights.
    The quantiles of a standard normal distribution are used to create a 4-bit representation
    with 16 values, normalized to the range [-1, 1].
    
    Math:
    Let Q be the set of 16 NF4 quantiles.
    For an input tensor X, we first normalize it by its absolute maximum:
        scale = max(abs(X))
        X_norm = X / scale
    Then each element in X_norm is mapped to the nearest value in Q:
        X_quant = argmin_{q in Q} abs(X_norm - q)
    Dequantization is simply:
        X_dequant = X_quant * scale
    """
    
    def __init__(self):
        super().__init__(bits=4)
        # NF4 values from QLoRA paper, normalized to [-1, 1]
        nf4_values = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ]
        # Store as a sorted tensor for efficient search
        self.register_buffer("quantiles", torch.tensor(nf4_values, dtype=torch.float32))
        
    def register_buffer(self, name, tensor):
        # Using simple attribute assignment since BaseQuantizer is not an nn.Module
        setattr(self, name, tensor)

    def quantize(self, x: torch.Tensor):
        """
        Quantizes the input tensor using NF4.
        Returns the quantized (but still float-typed) values mapped to the NF4 quantiles,
        and the scale factor used.
        """
        self.scale = x.abs().max()
        if self.scale == 0:
            return torch.zeros_like(x), self.scale
            
        x_norm = x / self.scale
        
        # Broadcasting to find the nearest quantile
        # x_norm: [..., 1], quantiles: [16]
        diffs = torch.abs(x_norm.unsqueeze(-1) - self.quantiles.to(x.device))
        indices = torch.argmin(diffs, dim=-1)
        
        # Map indices back to quantile values
        qx = self.quantiles.to(x.device)[indices]
        return qx, self.scale

    def dequantize(self, qx: torch.Tensor, scale: torch.Tensor):
        """
        Dequantizes the NF4 representation back to original scale.
        """
        return qx * scale


class FP4Quantizer(BaseQuantizer):
    """
    Float4 (FP4) Quantizer (E2M1).
    
    This implements a 4-bit floating point format with 1 sign bit, 2 exponent bits, 
    and 1 mantissa bit (E2M1), often used in LLM quantization.
    
    Math:
    The standard E2M1 representation provides 16 values. When normalized to [-1, 1],
    these values represent points linearly or logarithmically spaced depending on the 
    exponent. The positive values (before normalization) are typically:
    0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    Normalized to [-1, 1] by dividing by 6.0, we get the FP4 quantiles.
    
    Quantization finds the nearest FP4 value for absolute-max normalized input.
    """
    
    def __init__(self):
        super().__init__(bits=4)
        # E2M1 positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        # Normalized by max value 6:
        pos_values = [0.0, 0.5/6, 1.0/6, 1.5/6, 2.0/6, 3.0/6, 4.0/6, 1.0]
        neg_values = [-v for v in reversed(pos_values[1:])]
        fp4_values = neg_values + pos_values
        
        self.register_buffer("quantiles", torch.tensor(fp4_values, dtype=torch.float32))
        
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def quantize(self, x: torch.Tensor):
        """
        Quantizes the input tensor using FP4.
        Returns the quantized values mapped to the FP4 quantiles, and the scale.
        """
        self.scale = x.abs().max()
        if self.scale == 0:
            return torch.zeros_like(x), self.scale
            
        x_norm = x / self.scale
        
        diffs = torch.abs(x_norm.unsqueeze(-1) - self.quantiles.to(x.device))
        indices = torch.argmin(diffs, dim=-1)
        
        qx = self.quantiles.to(x.device)[indices]
        return qx, self.scale

    def dequantize(self, qx: torch.Tensor, scale: torch.Tensor):
        """
        Dequantizes the FP4 representation back to original scale.
        """
        return qx * scale
