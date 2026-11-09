"""
Educational Demonstration & Benchmark: Group-wise Quantization in LLMs.

Demonstrates the numerical and memory advantages of Group-wise (Block-wise)
Quantization over coarse Per-Tensor and Per-Channel schemes on a realistic
Large Language Model projection matrix (e.g. LLaMA/Mistral FFN projection).

Key Takeaways:
- Outlier resilience: Isolating outliers to blocks of size G = 128 prevents
  dynamic range collapse across the rest of the channel.
- Memory compression: 4-bit group-wise weights achieve ~7.5x compression vs FP32
  with minimal reconstruction error and >0.999 output cosine similarity.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from inwhale.llm.groupwise import (
    GroupwiseQuantizer,
    GroupwiseLinear,
    groupwise_quantize,
    groupwise_dequantize,
    groupwise_fake_quantize,
)


def compute_snr_db(signal: torch.Tensor, noisy: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in decibels."""
    signal_power = torch.sum(signal ** 2)
    noise_power = torch.sum((signal - noisy) ** 2)
    if noise_power == 0:
        return float("inf")
    return (10.0 * torch.log10(signal_power / noise_power)).item()


def generate_llm_synthetic_weights(
    out_features: int = 1024,
    in_features: int = 4096,
    outlier_fraction: float = 0.001,
    outlier_magnitude: float = 6.0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate synthetic weight matrix modeling empirical LLM weight distributions:
    99.9% Gaussian noise N(0, 0.02^2) and 0.1% kurtic outliers (Dettmers et al., 2022).
    """
    torch.manual_seed(seed)
    weights = torch.randn(out_features, in_features) * 0.02

    # Inject sparse high-magnitude outliers in specific feature channels
    num_outliers = int(out_features * in_features * outlier_fraction)
    outlier_rows = torch.randint(0, out_features, (num_outliers,))
    outlier_cols = torch.randint(0, in_features, (num_outliers,))
    outlier_signs = torch.sign(torch.randn(num_outliers))
    outlier_signs[outlier_signs == 0] = 1.0

    weights[outlier_rows, outlier_cols] = outlier_signs * outlier_magnitude
    return weights


def run_benchmark():
    print("=" * 80)
    print("   INWHALE: GROUP-WISE QUANTIZATION BENCHMARK FOR LARGE LANGUAGE MODELS   ")
    print("=" * 80)

    out_features = 1024
    in_features = 4096
    num_params = out_features * in_features
    fp32_mb = (num_params * 4.0) / (1024 * 1024)

    print(f"\nModel Configuration:")
    print(f"  Linear Layer:      {in_features} -> {out_features} (e.g. LLaMA MLP Gate/Up)")
    print(f"  Parameter Count:   {num_params:,} parameters")
    print(f"  FP32 Memory:       {fp32_mb:.2f} MB")
    print(f"  Outlier Profile:   0.1% extreme outliers (+/- 6.0) across feature channels")

    weights = generate_llm_synthetic_weights(out_features, in_features)
    test_input = torch.randn(8, in_features)  # Batch of 8 activation vectors
    golden_output = F.linear(test_input, weights)

    # 1. Per-Tensor INT4 Baseline
    t0 = time.perf_counter()
    scale_pt = weights.abs().max() / 7.0
    qx_pt = torch.clamp(torch.round(weights / scale_pt), -8, 7)
    rec_pt = qx_pt * scale_pt
    dt_pt = (time.perf_counter() - t0) * 1000.0

    mse_pt = torch.mean((weights - rec_pt) ** 2).item()
    snr_pt = compute_snr_db(weights, rec_pt)
    cos_pt = F.cosine_similarity(golden_output.flatten(), F.linear(test_input, rec_pt).flatten(), dim=0).item()
    mem_pt = (num_params * 4 / 8.0 + 4.0) / (1024 * 1024)

    results = [{
        "name": "Per-Tensor INT4",
        "mse": mse_pt,
        "snr_db": snr_pt,
        "cos_sim": cos_pt,
        "mem_mb": mem_pt,
        "compression": fp32_mb / mem_pt,
        "latency_ms": dt_pt,
    }]

    # 2. Group-wise configurations
    gw_configs = [
        ("Per-Channel INT4", in_features),
        ("Group-wise INT4 (G=256)", 256),
        ("Group-wise INT4 (G=128)", 128),
        ("Group-wise INT4 (G=64)", 64),
        ("Group-wise INT4 (G=32)", 32),
    ]

    for name, g_size in gw_configs:
        t0 = time.perf_counter()
        qx, scale, zp, pad_len = groupwise_quantize(
            x=weights,
            bits=4,
            group_size=g_size,
            dim=1,
            symmetric=True,
        )
        rec_weights = groupwise_dequantize(
            qx=qx,
            scale=scale,
            zero_point=zp,
            group_size=g_size,
            dim=1,
            pad_len=pad_len,
        )
        dt = (time.perf_counter() - t0) * 1000.0

        mse = torch.mean((weights - rec_weights) ** 2).item()
        snr = compute_snr_db(weights, rec_weights)
        quant_output = F.linear(test_input, rec_weights)
        cos_sim = F.cosine_similarity(golden_output.flatten(), quant_output.flatten(), dim=0).item()

        num_groups = (in_features + g_size - 1) // g_size
        total_groups = out_features * num_groups
        mem_bytes = (num_params * 4) / 8.0 + (total_groups * 4.0)
        mem_mb = mem_bytes / (1024 * 1024)
        compression = fp32_mb / mem_mb

        results.append({
            "name": name,
            "mse": mse,
            "snr_db": snr,
            "cos_sim": cos_sim,
            "mem_mb": mem_mb,
            "compression": compression,
            "latency_ms": dt,
        })

    print("\nBenchmark Results:")
    print("-" * 88)
    print(f"{'Quantization Scheme':<26} | {'MSE':<10} | {'SNR (dB)':<9} | {'Cos Sim':<8} | {'Memory':<8} | {'Ratio':<6}")
    print("-" * 88)

    for r in results:
        print(
            f"{r['name']:<26} | {r['mse']:<10.6f} | {r['snr_db']:<9.2f} | "
            f"{r['cos_sim']:<8.4f} | {r['mem_mb']:>5.2f} MB | {r['compression']:>5.1f}x"
        )
    print("-" * 88)

    print("\n--- Drop-in GroupwiseLinear Module Demonstration ---")
    base_linear = nn.Linear(in_features, out_features, bias=True)
    base_linear.weight.data.copy_(weights)

    # Convert to GroupwiseLinear (INT4, group_size=128)
    gw_layer = GroupwiseLinear.from_linear(base_linear, bits=4, group_size=128, symmetric=True)
    out_gw = gw_layer(test_input)
    base_out = base_linear(test_input)

    diff = torch.abs(out_gw - base_out).max().item()
    savings = gw_layer.memory_savings()

    print(f"Instantiated: GroupwiseLinear(in={in_features}, out={out_features}, bits=4, group_size=128)")
    print(f"FP32 Buffer Size:        {savings['fp32_bytes'] / (1024 * 1024):.2f} MB")
    print(f"Compressed Buffer Size:  {savings['quantized_bytes'] / (1024 * 1024):.2f} MB")
    print(f"Effective Compression:   {savings['compression_ratio']:.2f}x reduction")
    print(f"Max Absolute Error:      {diff:.4f}")
    print(f"Output Cosine Match:     {F.cosine_similarity(base_out.flatten(), out_gw.flatten(), dim=0).item():.5f}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
