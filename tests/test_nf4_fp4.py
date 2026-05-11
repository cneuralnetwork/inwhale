import torch
import pytest

from inwhale.llm.formats import NF4Quantizer, FP4Quantizer

def test_nf4_quantizer():
    # Setup
    quantizer = NF4Quantizer()
    
    # Simple input
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    
    # Process
    qx, scale = quantizer.quantize(x)
    x_dequant = quantizer.dequantize(qx, scale)
    
    # Assertions
    # Scale should be max(abs(x)) = 2.0
    assert scale.item() == 2.0
    
    # Max value 2.0 should map to max NF4 quantile 1.0, then back to 2.0
    # Min value -2.0 should map to min NF4 quantile -1.0, then back to -2.0
    assert torch.isclose(x_dequant[0], torch.tensor(-2.0))
    assert torch.isclose(x_dequant[-1], torch.tensor(2.0))
    
    # Lengths should match
    assert qx.shape == x.shape
    assert x_dequant.shape == x.shape


def test_fp4_quantizer():
    # Setup
    quantizer = FP4Quantizer()
    
    # Simple input
    x = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=torch.float32)
    
    # Process
    qx, scale = quantizer.quantize(x)
    x_dequant = quantizer.dequantize(qx, scale)
    
    # Assertions
    # Scale should be max(abs(x)) = 3.0
    assert scale.item() == 3.0
    
    # Max value 3.0 should map to max FP4 quantile 1.0, then back to 3.0
    assert torch.isclose(x_dequant[0], torch.tensor(-3.0))
    assert torch.isclose(x_dequant[-1], torch.tensor(3.0))


def test_nf4_zero_tensor():
    # Test edge case with all zeros
    quantizer = NF4Quantizer()
    x = torch.zeros(5, dtype=torch.float32)
    qx, scale = quantizer.quantize(x)
    x_dequant = quantizer.dequantize(qx, scale)
    
    assert scale.item() == 0.0
    assert torch.all(x_dequant == 0.0)

def test_fp4_zero_tensor():
    # Test edge case with all zeros
    quantizer = FP4Quantizer()
    x = torch.zeros(5, dtype=torch.float32)
    qx, scale = quantizer.quantize(x)
    x_dequant = quantizer.dequantize(qx, scale)
    
    assert scale.item() == 0.0
    assert torch.all(x_dequant == 0.0)
