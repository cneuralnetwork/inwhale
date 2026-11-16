"""
Demonstration & Benchmark: Mixed Precision Activations (8-bit Base + 16-bit Outliers) for LLMs.

Category 7: LLM-Specific Quantization (Milestone 7).

Overview:
---------
In Large Language Models (e.g. OPT, BLOOM, LLaMA, Mistral), activations at scale
exhibit emergent systematic outliers: a tiny fraction of hidden channels (< 0.1%)
features activation magnitudes 10x-100x larger than normal channels (e.g. |X| >= 6.0).

This script demonstrates:
1. The Failure Mode of Standard INT8:
   Standard INT8 quantization scale s = max(|X|) / 127 is stretched by extreme outliers,
   crushing >99.9% of normal activations to 0 or +/-1 (precision collapse).
2. The LLM.int8() Solution:
   Decomposes activations into:
   - 16-bit floating point path for outlier channels (|X| >= threshold).
   - 8-bit INT8 GEMM for regular channels (|X| < threshold).
3. Quantitative Comparison:
   Measures SNR (dB), Cosine Similarity, MAE, and memory footprints on a full-scale
   Transformer projection layer (4096 hidden dimension, 128 sequence length).
4. Drop-in Layer Integration:
   Demonstrates replacing nn.Linear with MixedPrecisionLinear.
"""

import math
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from inwhale.llm.mixed_precision import (
    find_outlier_channels,
    quantize_int8_vector_wise,
    dequantize_int8_vector_wise,
    mixed_precision_matmul,
    MixedPrecisionActivationQuantizer,
    MixedPrecisionLinear,
    compare_mixed_vs_uniform_8bit,
)


def run_demo():
    print("=" * 80)
    print("INWHALE: Mixed Precision Activations (8-bit Base + 16-bit Outliers) for LLMs")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    torch.manual_seed(42)

    # ------------------------------------------------------------------------
    # 1. Simulate Realistic Transformer Projection with Emergent Outliers
    # ------------------------------------------------------------------------
    batch_size = 1
    seq_len = 128
    hidden_dim = 4096    # Standard LLaMA-7B / Mistral hidden dimension
    out_features = 4096  # Projection dimension

    print(f"\n[1] Generating Transformer Activation & Weight Tensors:")
    print(f"    Activation Shape (Tokens x Dim) : ({seq_len}, {hidden_dim})")
    print(f"    Weight Matrix Shape (D x C)     : ({out_features}, {hidden_dim})")
    print(f"    Total Parameters                : {hidden_dim * out_features:,} ({hidden_dim * out_features * 4 / (1024**2):.1f} MB in FP32)")

    # Activations follow zero-mean standard Gaussian across normal channels
    x = torch.randn(seq_len, hidden_dim, device=device) * 0.4
    w = (torch.randn(out_features, hidden_dim, device=device) / math.sqrt(hidden_dim))
    bias = torch.zeros(out_features, device=device)

    # Inject 7 systematic outlier channels (< 0.2% of channels), matching LLM.int8() empirical findings
    outlier_channel_indices = [128, 512, 1024, 1890, 2450, 3100, 3950]
    for ch in outlier_channel_indices:
        # Extreme magnitudes between 25.0 and 65.0
        sign = 1.0 if ch % 2 == 0 else -1.0
        x[:, ch] = sign * (25.0 + (ch % 40))

    threshold = 6.0
    detected = find_outlier_channels(x, threshold=threshold)
    print(f"\n[2] Outlier Analysis (Threshold = {threshold}):")
    print(f"    Detected Outlier Channels : {detected.cpu().tolist()}")
    print(f"    Outlier Channel Count     : {detected.numel()} / {hidden_dim} ({detected.numel() / hidden_dim * 100:.3f}% of channels)")
    print(f"    Normal Feature Channels   : {hidden_dim - detected.numel()} ({100 - detected.numel() / hidden_dim * 100:.3f}%)")
    print(f"    Peak Activation Value     : {torch.max(torch.abs(x)).item():.2f}")
    print(f"    Normal Channel Max Value  : {torch.max(torch.abs(x[:, ~torch.isin(torch.arange(hidden_dim, device=device), detected)])).item():.2f}")

    # ------------------------------------------------------------------------
    # 2. Benchmark Full Precision vs Pure INT8 vs Mixed Precision
    # ------------------------------------------------------------------------
    print("\n[3] Comparing Output Fidelity vs FP32 Ground Truth:")
    stats = compare_mixed_vs_uniform_8bit(x, w, threshold=threshold, vector_wise=True)

    print("-" * 80)
    print(f"{'Quantization Scheme':<35} | {'SNR (dB)':<10} | {'Cosine Sim':<12} | {'MAE':<10} | {'Max Err':<10}")
    print("-" * 80)
    print(f"{'1. Standard Pure INT8 (Vector-wise)':<35} | {stats['pure_int8_snr_db']:<10.2f} | {stats['pure_int8_cosine_sim']:<12.6f} | {stats['pure_int8_mae']:<10.4f} | {stats['pure_int8_max_err']:<10.4f}")
    print(f"{'2. Mixed Precision (8-bit + 16-bit)':<35} | {stats['mixed_precision_snr_db']:<10.2f} | {stats['mixed_precision_cosine_sim']:<12.6f} | {stats['mixed_precision_mae']:<10.4f} | {stats['mixed_precision_max_err']:<10.4f}")
    print("-" * 80)
    print(f"==> SNR Fidelity Improvement: +{stats['snr_improvement_db']:.2f} dB higher fidelity with Mixed Precision!")
    print(f"==> Relative Error Reduction: {(1.0 - stats['mixed_precision_mae'] / max(stats['pure_int8_mae'], 1e-8)) * 100:.1f}% lower MAE")

    # ------------------------------------------------------------------------
    # 3. Object-Oriented Quantizer Demonstration
    # ------------------------------------------------------------------------
    print("\n[4] Object-Oriented MixedPrecisionActivationQuantizer Demo:")
    quantizer = MixedPrecisionActivationQuantizer(bits=8, threshold=threshold, outlier_dtype=torch.float16)
    state = quantizer.quantize(x)
    print(f"    Decomposed State:")
    print(f"      - 16-bit Outliers Shape      : {tuple(state.outliers.shape)} in {state.outliers.dtype}")
    print(f"      - 8-bit Quantized Regular    : {tuple(state.quantized_regular.shape)} in {state.quantized_regular.dtype}")
    print(f"      - Regular Scales Shape       : {tuple(state.regular_scales.shape)} in {state.regular_scales.dtype}")

    x_recon = quantizer.dequantize(state)
    recon_cos = F.cosine_similarity(x.flatten().unsqueeze(0), x_recon.flatten().unsqueeze(0)).item()
    print(f"    Tensor Reconstruction Cosine Fidelity: {recon_cos:.7f}")

    # ------------------------------------------------------------------------
    # 4. Drop-in MixedPrecisionLinear Layer Demonstration
    # ------------------------------------------------------------------------
    print("\n[5] Drop-in PyTorch Module (MixedPrecisionLinear):")
    base_linear = nn.Linear(hidden_dim, out_features, bias=True, device=device)
    base_linear.weight.data.copy_(w)
    base_linear.bias.data.copy_(bias)

    # Convert to MixedPrecisionLinear
    mp_linear = MixedPrecisionLinear.from_linear(base_linear, threshold=threshold, outlier_dtype=torch.float16)
    print(f"    Layer Representation: {mp_linear}")

    mem_stats = mp_linear.memory_stats()
    print(f"    Memory Footprint:")
    print(f"      - Full Precision (FP32)    : {mem_stats['weight_bytes'] / (1024**2):.2f} MB")
    print(f"      - Quantized Representation : {mem_stats['theoretical_int8_bytes'] / (1024**2):.2f} MB")
    print(f"      - Compression Ratio        : {mem_stats['compression_ratio']:.2f}x reduction")

    # Forward pass
    mp_linear.eval()
    with torch.no_grad():
        y_ref = base_linear(x)
        y_mp = mp_linear(x)

    layer_cos = F.cosine_similarity(y_ref.flatten().unsqueeze(0), y_mp.flatten().unsqueeze(0)).item()
    layer_snr = 10.0 * math.log10(torch.sum(y_ref**2).item() / max(torch.sum((y_ref - y_mp)**2).item(), 1e-12))
    print(f"    Layer Output Fidelity vs nn.Linear:")
    print(f"      - Cosine Similarity : {layer_cos:.6f}")
    print(f"      - Output SNR (dB)   : {layer_snr:.2f} dB")

    print("\n" + "=" * 80)
    print("Demonstration successfully completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
