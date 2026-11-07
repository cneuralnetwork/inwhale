"""
Trained Ternary Quantization (TTQ) Demonstration.

This example illustrates:
1. Quantizing full-precision weights into ternary states {-wn, 0, +wp}.
2. Asymmetric scaling factors (wp != wn) to preserve representation quality.
3. Memory and sparsity advantages (ternary codes require only 2 bits).
4. Quantization-Aware Training (QAT) with TTQLinear and Straight-Through Estimation.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from inwhale.qat.ttq import TTQLinear, TTQQuantizer, ttq_quantize

print("=" * 70)
print("  TRAINED TERNARY QUANTIZATION (TTQ) - INWHALE DEMO")
print("=" * 70)

# Section 1: Basic TTQ Quantization & Ternary Discretization
print("\n[Section 1] Weight Quantization & Sparsity Analysis")
torch.manual_seed(42)

# Simulated neural network weight slice (FP32)
weights = torch.tensor(
    [
        [-0.85, -0.42, -0.03,  0.02,  0.31,  0.92],
        [-0.60, -0.15, -0.01,  0.04,  0.55,  0.78],
        [-0.30, -0.04,  0.00,  0.03,  0.20,  0.45],
    ]
)

quantizer = TTQQuantizer(threshold_ratio=0.05)
codes = quantizer.quantize(weights)
dequantized = quantizer.dequantize(codes)

delta = quantizer.delta.item()
wp = quantizer.wp.item()
wn = quantizer.wn.item()

total_params = weights.numel()
zero_params = (codes == 0).sum().item()
pos_params = (codes == 1).sum().item()
neg_params = (codes == -1).sum().item()
sparsity = (zero_params / total_params) * 100.0

print(f"Original FP32 weights (shape {weights.shape}):")
print(weights)
print(f"\nThreshold Delta: {delta:.4f} (5% of max |W| = {weights.abs().max():.4f})")
print(f"Learned Positive Scale (wp): {wp:.4f}")
print(f"Learned Negative Scale (wn): {wn:.4f}")
print(f"\nTernary Codes Q in {{-1, 0, +1}} (2 bits per weight):")
print(codes)
print(f"\nDequantized Reconstructed Weights in {{-wn, 0, +wp}}:")
print(dequantized)

print("\n--- Compression & Sparsity Statistics ---")
print(f"Total Parameters:    {total_params}")
print(f"Positive Weights:    {pos_params} ({pos_params/total_params*100:.1f}%) -> +{wp:.4f}")
print(f"Negative Weights:    {neg_params} ({neg_params/total_params*100:.1f}%) -> -{wn:.4f}")
print(f"Zero / Pruned:       {zero_params} ({sparsity:.1f}%) -> 0.0000")
print(f"FP32 Storage:        {total_params * 32} bits")
print(f"Ternary Storage:     {total_params * 2} bits (16x memory compression)")

# Section 2: Quantization-Aware Training (QAT) with TTQLinear
print("\n" + "=" * 70)
print("[Section 2] Quantization-Aware Training (QAT) with TTQLinear")
print("=" * 70)

# Generate synthetic non-linear classification dataset
torch.manual_seed(123)
N = 200
D_in = 8
D_hidden = 16
D_out = 1

X = torch.randn(N, D_in)
# Ground truth non-linear decision boundary
Y = ((X[:, :3].sum(dim=1) + 0.5 * X[:, 3] * X[:, 4]) > 0).float().unsqueeze(1)

# Build a TTQ model with ternary linear layers
model = nn.Sequential(
    TTQLinear(D_in, D_hidden, bias=True, threshold_ratio=0.05),
    nn.ReLU(),
    TTQLinear(D_hidden, D_out, bias=True, threshold_ratio=0.05),
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.04)

print("\nTraining TTQ network with Straight-Through Estimation...")
for epoch in range(1, 26):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, Y)
    loss.backward()
    optimizer.step()

    # Enforce non-negative constraint on scale factors
    for module in model.modules():
        if isinstance(module, TTQLinear):
            module.clamp_scales(min_val=1e-4)

    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == Y).float().mean() * 100.0

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:2d}/25 | BCE Loss: {loss.item():.4f} | Accuracy: {acc.item():.1f}%")

print("\nLayer 1 Trained Scales:")
l1: TTQLinear = model[0]
print(f"  wp: {l1.wp.item():.4f}, wn: {l1.wn.item():.4f}")
l1_q = l1.get_quantized_weight()
l1_zeros = (l1_q == 0).sum().item()
print(f"  Layer 1 sparsity: {l1_zeros}/{l1_q.numel()} weights ({l1_zeros/l1_q.numel()*100:.1f}%) pruned to zero.")

print("\nLayer 2 Trained Scales:")
l2: TTQLinear = model[2]
print(f"  wp: {l2.wp.item():.4f}, wn: {l2.wn.item():.4f}")
l2_q = l2.get_quantized_weight()
l2_zeros = (l2_q == 0).sum().item()
print(f"  Layer 2 sparsity: {l2_zeros}/{l2_q.numel()} weights ({l2_zeros/l2_q.numel()*100:.1f}%) pruned to zero.")

print("\nDemo completed successfully!")
