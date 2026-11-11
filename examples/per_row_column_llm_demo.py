"""
Per-Row and Per-Column Quantization for Large Language Models (LLMs) Demonstration.

Category 7: LLM-Specific Quantization (Milestone 7).

This script simulates a realistic Transformer projection layer (4.19M parameters)
with heavy-tailed activation outliers, comparing:
1. Per-Tensor INT8 Quantization
2. Per-Row INT8 Quantization
3. Per-Column INT8 Quantization
4. Vector-Wise INT8 GEMM (LLM.int8() style)
5. Per-Row INT4 Quantization
6. Per-Column INT4 Quantization

Metrics:
- Mean Squared Error (MSE)
- Signal-to-Noise Ratio (SNR in dB)
- Output Cosine Similarity
- Memory Footprint & Compression Ratio
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from inwhale.llm.per_row_column import (
    per_row_quantize,
    per_row_dequantize,
    per_column_quantize,
    per_column_dequantize,
    vectorwise_quantized_matmul,
    PerRowLinear,
    PerColumnLinear,
)


def compute_snr(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in decibels (dB)."""
    signal_power = torch.mean(orig**2).item()
    noise_power = torch.mean((orig - recon) ** 2).item()
    if noise_power <= 1e-12:
        return 100.0
    return 10.0 * math.log10(max(signal_power / noise_power, 1e-12))


def run_benchmark():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 74)
    print(f" LLM QUANTIZATION BENCHMARK: PER-ROW & PER-COLUMN (Device: {device})")
    print("=" * 74)

    # Simulation setup: LLaMA / Mistral MLP projection matrix
    d_in = 4096
    d_out = 1024
    num_tokens = 64
    num_weights = d_in * d_out
    fp32_mb = (num_weights * 4.0) / (1024 * 1024)

    print(f"Model Layer: Transformer Linear Projection ({d_in} -> {d_out})")
    print(f"Total Parameters: {num_weights:,} weights ({fp32_mb:.2f} MB in FP32)")
    print(f"Batch Tokens: {num_tokens} tokens")

    # Generate synthetic weights with realistic kurtosis and channel outliers
    torch.manual_seed(42)
    weight = torch.randn(d_out, d_in, device=device) * 0.02
    # Add systematic channel-aligned outliers (Dettmers et al. 2022)
    outlier_channels = [12, 105, 512, 2048]
    for ch in outlier_channels:
        weight[:, ch] *= 25.0  # 25x Kurtic outlier channels

    # Generate token activations
    activation = torch.randn(num_tokens, d_in, device=device)
    # Add token-specific outliers (e.g., initial prompt token, delimiter tokens)
    activation[0, :] *= 10.0

    # FP32 reference projection: Y = X @ W^T
    ref_output = F.linear(activation, weight)

    results = []

    # 1. Per-Tensor INT8
    max_tensor = torch.max(torch.abs(weight))
    s_tensor = max_tensor / 127.0
    q_tensor = torch.clamp(torch.round(weight / s_tensor), -128, 127)
    rec_tensor = q_tensor * s_tensor
    out_tensor = F.linear(activation, rec_tensor)

    mse_pt8 = torch.mean((ref_output - out_tensor) ** 2).item()
    snr_pt8 = compute_snr(ref_output, out_tensor)
    cos_pt8 = torch.cosine_similarity(
        ref_output.flatten(), out_tensor.flatten(), dim=0
    ).item()
    mem_pt8 = (num_weights * 1.0 + 4.0) / (1024 * 1024)
    results.append(
        ("Per-Tensor INT8", mse_pt8, snr_pt8, cos_pt8, mem_pt8, fp32_mb / mem_pt8)
    )

    # 2. Per-Row INT8
    qx_row8, s_row8, _ = per_row_quantize(weight, bits=8, symmetric=True)
    rec_row8 = per_row_dequantize(qx_row8, s_row8)
    out_row8 = F.linear(activation, rec_row8)

    mse_pr8 = torch.mean((ref_output - out_row8) ** 2).item()
    snr_pr8 = compute_snr(ref_output, out_row8)
    cos_pr8 = torch.cosine_similarity(
        ref_output.flatten(), out_row8.flatten(), dim=0
    ).item()
    mem_pr8 = (num_weights * 1.0 + d_out * 4.0) / (1024 * 1024)
    results.append(
        ("Per-Row INT8", mse_pr8, snr_pr8, cos_pr8, mem_pr8, fp32_mb / mem_pr8)
    )

    # 3. Per-Column INT8
    qx_col8, s_col8, _ = per_column_quantize(weight, bits=8, symmetric=True)
    rec_col8 = per_column_dequantize(qx_col8, s_col8)
    out_col8 = F.linear(activation, rec_col8)

    mse_pc8 = torch.mean((ref_output - out_col8) ** 2).item()
    snr_pc8 = compute_snr(ref_output, out_col8)
    cos_pc8 = torch.cosine_similarity(
        ref_output.flatten(), out_col8.flatten(), dim=0
    ).item()
    mem_pc8 = (num_weights * 1.0 + d_in * 4.0) / (1024 * 1024)
    results.append(
        ("Per-Column INT8", mse_pc8, snr_pc8, cos_pc8, mem_pc8, fp32_mb / mem_pc8)
    )

    # 4. Vector-Wise INT8 GEMM (Per-Row Token Activation + Per-Column Weight)
    out_vw8 = vectorwise_quantized_matmul(
        activation=activation,
        weight=weight,
        bits=8,
        transpose_weight=True,
    )
    mse_vw8 = torch.mean((ref_output - out_vw8) ** 2).item()
    snr_vw8 = compute_snr(ref_output, out_vw8)
    cos_vw8 = torch.cosine_similarity(
        ref_output.flatten(), out_vw8.flatten(), dim=0
    ).item()
    results.append(
        ("Vector-wise INT8 GEMM", mse_vw8, snr_vw8, cos_vw8, mem_pc8, fp32_mb / mem_pc8)
    )

    # 5. Per-Row INT4
    qx_row4, s_row4, _ = per_row_quantize(weight, bits=4, symmetric=True)
    rec_row4 = per_row_dequantize(qx_row4, s_row4)
    out_row4 = F.linear(activation, rec_row4)

    mse_pr4 = torch.mean((ref_output - out_row4) ** 2).item()
    snr_pr4 = compute_snr(ref_output, out_row4)
    cos_pr4 = torch.cosine_similarity(
        ref_output.flatten(), out_row4.flatten(), dim=0
    ).item()
    mem_pr4 = (num_weights * 0.5 + d_out * 4.0) / (1024 * 1024)
    results.append(
        ("Per-Row INT4", mse_pr4, snr_pr4, cos_pr4, mem_pr4, fp32_mb / mem_pr4)
    )

    # 6. Per-Column INT4
    qx_col4, s_col4, _ = per_column_quantize(weight, bits=4, symmetric=True)
    rec_col4 = per_column_dequantize(qx_col4, s_col4)
    out_col4 = F.linear(activation, rec_col4)

    mse_pc4 = torch.mean((ref_output - out_col4) ** 2).item()
    snr_pc4 = compute_snr(ref_output, out_col4)
    cos_pc4 = torch.cosine_similarity(
        ref_output.flatten(), out_col4.flatten(), dim=0
    ).item()
    mem_pc4 = (num_weights * 0.5 + d_in * 4.0) / (1024 * 1024)
    results.append(
        ("Per-Column INT4", mse_pc4, snr_pc4, cos_pc4, mem_pc4, fp32_mb / mem_pc4)
    )

    # Print results table
    print("\n" + "=" * 80)
    print(
        f"{'Quantization Scheme':<26} | {'MSE':<10} | {'SNR (dB)':<9} | {'Cosine Sim':<11} | {'Mem (MB)':<9} | {'Speedup/Ratio'}"
    )
    print("-" * 80)
    for name, mse, snr, cos_sim, mem, ratio in results:
        print(
            f"{name:<26} | {mse:<10.6f} | {snr:>7.2f} dB | {cos_sim:>10.5f}  | {mem:>6.2f} MB | {ratio:>5.2f}x"
        )
    print("=" * 80)

    # Verification of PerRowLinear and PerColumnLinear drop-in layers
    print("\nVerifying PyTorch Drop-in Modules (PerRowLinear & PerColumnLinear)...")
    linear_orig = nn.Linear(d_in, d_out).to(device)
    row_module = PerRowLinear.from_linear(linear_orig, bits=8).to(device)
    col_module = PerColumnLinear.from_linear(linear_orig, bits=8).to(device)

    y_orig = linear_orig(activation)
    y_row = row_module(activation)
    y_col = col_module(activation)

    cos_row = torch.cosine_similarity(y_orig.flatten(), y_row.flatten(), dim=0).item()
    cos_col = torch.cosine_similarity(y_orig.flatten(), y_col.flatten(), dim=0).item()

    print(
        f"PerRowLinear Cosine Fidelity:    {cos_row:.5f} (Compression: {row_module.memory_savings()['compression_ratio']:.2f}x)"
    )
    print(
        f"PerColumnLinear Cosine Fidelity: {cos_col:.5f} (Compression: {col_module.memory_savings()['compression_ratio']:.2f}x)"
    )
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
