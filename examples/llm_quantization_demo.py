import torch
import matplotlib.pyplot as plt
from inwhale.llm.formats import NF4Quantizer, FP4Quantizer

def demo_llm_quantization():
    print("=== NF4 and FP4 Quantization Demo ===")
    
    # 1. Create normally distributed "weights"
    torch.manual_seed(42)
    weights = torch.randn(1000)
    
    print(f"\nOriginal Weights:")
    print(f"Shape: {weights.shape}, Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")
    
    # 2. Initialize the quantizers
    nf4_quantizer = NF4Quantizer()
    fp4_quantizer = FP4Quantizer()
    
    # 3. Quantize the weights
    nf4_q, nf4_scale = nf4_quantizer.quantize(weights)
    fp4_q, fp4_scale = fp4_quantizer.quantize(weights)
    
    # 4. Dequantize
    nf4_deq = nf4_quantizer.dequantize(nf4_q, nf4_scale)
    fp4_deq = fp4_quantizer.dequantize(fp4_q, fp4_scale)
    
    # 5. Compute Error
    nf4_error = torch.nn.functional.mse_loss(weights, nf4_deq)
    fp4_error = torch.nn.functional.mse_loss(weights, fp4_deq)
    
    print("\nQuantization Errors (MSE vs Original):")
    print(f"NF4 Error: {nf4_error.item():.6f}")
    print(f"FP4 Error: {fp4_error.item():.6f}")
    
    # Notice that NF4 has lower MSE error for normally distributed weights
    # compared to FP4, which is the key finding from the QLoRA paper.
    print(f"\nNF4 is {'better' if nf4_error < fp4_error else 'worse'} than FP4 for normal distribution by {(fp4_error - nf4_error).abs().item():.6f} MSE.")
    
    # Show values distribution
    print("\nFirst 10 Weights comparison:")
    print("Original |   NF4   |   FP4")
    print("-" * 35)
    for i in range(10):
        print(f"{weights[i].item():>8.4f} | {nf4_deq[i].item():>7.4f} | {fp4_deq[i].item():>7.4f}")

if __name__ == "__main__":
    demo_llm_quantization()
