"""
Demonstration & Benchmark: Outlier-Aware Activation Quantization (SmoothQuant) for LLMs.

Category 7: LLM-Specific Quantization (Milestone 7).

Overview:
---------
In Large Language Models (e.g. OPT, BLOOM, LLaMA, Mistral), activations at scale
exhibit emergent systematic channel outliers: a tiny fraction of hidden channels
(< 0.2%) features activation magnitudes 10x - 100x larger than normal channels
(e.g. |X| >= 50.0 vs |X| < 1.0).

This script demonstrates:
1. Emergent Outlier Characteristics:
   - Diagnostic analysis of activation kurtosis and peak-to-average ratio (PAR).
   - Visualizing how extreme outlier channels distort the uniform INT8 dynamic range.
2. The SmoothQuant Solution (Xiao et al., ICML 2023):
   - Invertible diagonal channel migration: Y = (X * diag(s)^-1) * (diag(s) * W^T).
   - Multi-batch calibration via `SmoothQuantObserver`.
   - Optimal migration scale derivation with alpha = 0.5.
3. Zero-Overhead LayerNorm Folding:
   - Folding smoothing scales into preceding LayerNorm parameters offline.
4. Quantitative Comparison on Full-Scale Transformer Projection (4096 x 4096, 16.7M params):
   - Evaluates Output SNR (dB), Cosine Similarity, MAE, Max Error, and Memory Footprint
     across FP32, Standard INT8 (per-tensor & per-token), and SmoothQuant W8A8.
5. Drop-in Replacement:
   - Converting `nn.Linear` to `SmoothQuantLinear`.
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

from inwhale.llm.outlier_aware import (
    compute_smoothquant_scales,
    apply_activation_smoothing,
    apply_weight_smoothing,
    fold_smoothing_into_layernorm,
    analyze_activation_outliers,
    SmoothQuantObserver,
    smoothquant_gemm,
    SmoothQuantLinear,
    compare_smoothquant_vs_standard_8bit,
)


def run_demo():
    print("=" * 82)
    print("INWHALE: Outlier-Aware Activation Quantization (SmoothQuant) for LLMs")
    print("=" * 82)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Platform: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    torch.manual_seed(42)

    # ------------------------------------------------------------------------
    # 1. Simulate Scale & Outliers of a 7B LLM Transformer Layer
    # ------------------------------------------------------------------------
    batch_size = 8
    seq_len = 128
    hidden_dim = 4096  # LLaMA-7B / BLOOM-7B hidden size
    out_dim = 4096

    print(f"\n[1] Synthesizing Transformer Feed-Forward Projection Layer:")
    print(f"    - Activation Shape : ({batch_size}, {seq_len}, {hidden_dim}) [Tokens: {batch_size * seq_len}]")
    print(f"    - Weight Shape     : ({out_dim}, {hidden_dim}) [Parameters: {out_dim * hidden_dim:,}]")
    print(f"    - FP32 Memory      : {(out_dim * hidden_dim * 4) / (1024**2):.2f} MB")

    # Generate normal activation features with std = 0.4
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device) * 0.4

    # Inject emergent systematic outlier channels (matching empirical LLM observations)
    outlier_channels = [128, 512, 1024, 2048, 3072, 3840]
    for c in outlier_channels:
        x[:, :, c] += 65.0 * torch.sign(torch.randn(batch_size, seq_len, device=device))

    # Weight matrix initialized with realistic Xavier/Kaiming distribution
    weight = torch.randn(out_dim, hidden_dim, device=device) * (1.0 / math.sqrt(hidden_dim))
    bias = torch.zeros(out_dim, device=device)

    # ------------------------------------------------------------------------
    # 2. Diagnostic Analysis of Outlier Characteristics
    # ------------------------------------------------------------------------
    print(f"\n[2] Diagnostic Outlier Analysis:")
    diag = analyze_activation_outliers(x, threshold=10.0, dim=-1)
    print(f"    - Total Hidden Channels : {diag.num_channels}")
    print(f"    - Global Max Activation : {diag.global_max:.2f}")
    print(f"    - Global Mean Absolute  : {diag.global_mean:.4f}")
    print(f"    - Peak-to-Average Ratio : {diag.par:.2f}x")
    print(f"    - Channel Kurtosis      : {diag.kurtosis:.2f} (Gaussian baseline = 3.0)")
    print(f"    - Outlier Channel Count : {len(diag.outlier_channel_indices)} ({diag.outlier_ratio*100:.2f}%)")
    print(f"    - Outlier Coordinates   : {diag.outlier_channel_indices.tolist()[:10]}")

    # ------------------------------------------------------------------------
    # 3. Calibration & SmoothQuant Scale Migration
    # ------------------------------------------------------------------------
    print(f"\n[3] Calibration & Scale Migration Computation:")
    observer = SmoothQuantObserver(num_channels=hidden_dim)

    # Simulate multi-batch calibration
    print(f"    - Collecting statistics across 4 calibration batches...")
    for step in range(4):
        calib_batch = torch.randn(4, 64, hidden_dim, device=device) * 0.4
        for c in outlier_channels:
            calib_batch[:, :, c] += (60.0 + step * 2.0) * torch.sign(torch.randn(4, 64, device=device))
        observer.update(calib_batch)

    act_scales = observer.get_act_scales()
    scales_sq = observer.compute_scales(weight, alpha=0.5)

    print(f"    - SmoothQuant alpha     : 0.5 (Geometric mean migration)")
    print(f"    - Max Smoothing Scale s : {torch.max(scales_sq).item():.4f}")
    print(f"    - Min Smoothing Scale s : {torch.min(scales_sq).item():.4f}")

    # ------------------------------------------------------------------------
    # 4. Quantitative Benchmark: FP32 vs Standard INT8 vs SmoothQuant
    # ------------------------------------------------------------------------
    print(f"\n[4] Execution & Numerical Fidelity Benchmark:")
    results = compare_smoothquant_vs_standard_8bit(x, weight, bias=bias, alpha=0.5)

    print("-" * 82)
    print(f"{'Quantization Scheme':<36} | {'SNR (dB)':<10} | {'Cosine':<10} | {'MAE':<10} | {'Max Error':<10}")
    print("-" * 82)

    print(f"{'FP32 Full Precision Reference':<36} | {'Baseline':<10} | {'1.000000':<10} | {'0.000000':<10} | {'0.000000':<10}")

    for name, metrics in results.items():
        label = {
            "standard_int8_per_tensor": "Standard INT8 (Per-Tensor)",
            "standard_int8_per_token": "Standard INT8 (Per-Token W8A8)",
            "smoothquant_w8a8": "SmoothQuant INT8 (W8A8, alpha=0.5)",
        }.get(name, name)

        snr = f"{metrics['snr_db']:.2f} dB"
        cos = f"{metrics['cosine_similarity']:.6f}"
        mae = f"{metrics['mae']:.6f}"
        max_err = f"{metrics['max_error']:.6f}"
        print(f"{label:<36} | {snr:<10} | {cos:<10} | {mae:<10} | {max_err:<10}")

    print("-" * 82)

    snr_gain = results['smoothquant_w8a8']['snr_db'] - results['standard_int8_per_token']['snr_db']
    mae_reduction = (1.0 - (results['smoothquant_w8a8']['mae'] / results['standard_int8_per_token']['mae'])) * 100.0

    print(f"    * SmoothQuant SNR Improvement : +{snr_gain:.2f} dB over standard per-token INT8")
    print(f"    * Error Reduction             : {mae_reduction:.2f}% lower Mean Absolute Error")

    # ------------------------------------------------------------------------
    # 5. Drop-in Layer Integration & LayerNorm Folding
    # ------------------------------------------------------------------------
    print(f"\n[5] Drop-in PyTorch Module & Zero-Overhead LayerNorm Folding:")
    standard_linear = nn.Linear(hidden_dim, out_dim, bias=True, device=device)
    standard_linear.weight.data.copy_(weight)
    standard_linear.bias.data.copy_(bias)

    # Convert to SmoothQuantLinear
    sq_linear = SmoothQuantLinear.from_linear(
        standard_linear,
        act_scales=act_scales,
        alpha=0.5,
        quantize_weights=True,
    )
    sq_linear.eval()

    mem_stats = sq_linear.memory_stats()
    print(f"    - Full Precision Parameters   : {mem_stats['num_parameters']:,}")
    print(f"    - FP32 Memory Footprint       : {mem_stats['fp32_bytes'] / (1024**2):.2f} MB")
    print(f"    - INT8 Compressed Footprint   : {mem_stats['int8_total_bytes'] / (1024**2):.2f} MB")
    print(f"    - Compression Ratio           : {mem_stats['compression_ratio_vs_fp32']:.2f}x")

    # Demonstrate LayerNorm folding
    ln = nn.LayerNorm(hidden_dim, device=device)
    print(f"    - Folding smoothing scale vector into LayerNorm...")
    sq_linear.fold_into_layernorm(ln)
    print(f"    - LayerNorm folding status    : is_folded = {sq_linear.is_folded}")

    # Forward pass through folded pipeline
    start_t = time.perf_counter()
    with torch.no_grad():
        x_norm = ln(x)
        y_folded = sq_linear(x_norm)
    elapsed = (time.perf_counter() - start_t) * 1000.0

    print(f"    - Folded Forward Output Shape : {y_folded.shape}")
    print(f"    - Inference Latency           : {elapsed:.2f} ms")
    print("\n" + "=" * 82)
    print("SUCCESS: SmoothQuant outlier-aware activation quantization benchmark complete.")
    print("=" * 82)


if __name__ == "__main__":
    run_demo()
