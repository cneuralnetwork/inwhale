# Quantization Roadmap (TODO)

This file tracks planned and in-progress quantization implementations.
Each item represents a concrete, testable unit of work.

## Milestone 1: Uniform Quantization

Goal: establish numerical conventions, APIs, and reference behavior.

- [ ] Symmetric uniform quantizer
  - [ ] Scale computation
  - [ ] Clipping & saturation
  - [ ] Quantize / dequantize
  - [ ] Unit tests + demo

- [ ] Asymmetric uniform quantizer
  - [ ] Zero-point handling
  - [ ] Edge-case behavior
  - [ ] Tests

- [ ] Affine quantizer (scale + zero-point abstraction)

- [ ] Mid-rise quantizer
- [ ] Mid-tread quantizer
- [ ] Dead-zone quantizer

- [ ] Per-tensor quantization
- [ ] Per-channel quantization

- [ ] Mixed-precision uniform quantization
- [ ] Sub-byte quantization
  - [ ] 4-bit
  - [ ] 2-bit
  - [ ] 1-bit (binary)

## Milestone 2: Non-Uniform Quantization

Goal: implement non-linear mappings and codebook-based schemes.

- [ ] Logarithmic (power-of-two) quantization
- [ ] µ-law companding
- [ ] A-law companding

- [ ] LUT-based learned non-uniform quantizer
- [ ] Vector quantizer (basic k-means)
- [ ] Lattice quantizer

- [ ] Residual quantization
- [ ] Additive quantization

- [ ] Product quantization
  - [ ] PQ
  - [ ] OPQ
  - [ ] RQ

- [ ] VQ-VAE–style codebook quantization

## Milestone 3: Rounding Strategies

Goal: decouple rounding from quantization logic.

- [ ] Nearest rounding
- [ ] Stochastic rounding
- [ ] Banker’s rounding
- [ ] Round-away-from-zero
- [ ] Floor rounding
- [ ] Ceil rounding
- [ ] Truncation

- [ ] Adaptive rounding (AdaRound)
- [ ] Gradient-learned rounding (LSQ+ style)

## Milestone 4: Observers & Range Estimation

Goal: support PTQ and QAT scale estimation.

- [ ] MinMax observer
- [ ] Moving average observer
- [ ] Percentile observer

- [ ] MSE-based scale search
- [ ] KL-divergence observer
- [ ] Histogram-based observer

- [ ] Per-channel observers
- [ ] Gradient-based scale learning (LSQ family)

## Milestone 5: Post-Training Quantization (PTQ)

Goal: end-to-end PTQ pipelines.

- [ ] Static PTQ
- [ ] Dynamic PTQ
- [ ] Weight-only quantization

- [ ] SmoothQuant
- [ ] GPTQ (second-order PTQ)
- [ ] AWQ (outlier-aware PTQ)
- [ ] ZeroQuant

- [ ] Relaxed quantization (RQ)
- [ ] Outlier splitting (LLM.int8-style)
- [ ] Blockwise quantization

## Milestone 6: Quantization-Aware Training (QAT)

Goal: training-time quantization simulation.

- [ ] Fake quantization modules
- [ ] LSQ (Learned Step Size Quantization)
- [ ] LSQ+

- [ ] INQ (Incremental Quantization)
- [ ] DoReFa-Net

- [ ] Ternary quantization (TTQ)
- [ ] Binary quantization
  - [ ] XNOR-Net
  - [ ] BinaryConnect
  - [ ] ABC-Net

- [ ] QAT support for transformers and attention modules

## Milestone 7: LLM-Specific Quantization

Goal: practical large-model compression techniques.

- [ ] NF4 format
- [ ] FP4 format

- [x] Group-wise quantization
- [ ] Per-row quantization
- [ ] Per-column quantization

- [ ] Attention-specific quantization
- [ ] Outlier-aware activation quantization

- [ ] Quantized LoRA (QLoRA)
- [ ] Mixed-precision activations
  - [ ] 8-bit base
  - [ ] 16-bit outliers

## Milestone 8: Vector & Codebook Quantization (Advanced)

- [ ] PQ / OPQ / RQ / AQ unification
- [ ] Neural codebook learning
- [ ] Multi-level vector quantization

## Milestone 9: Hardware-Inspired Quantization

Goal: hardware-aligned numerical constraints.

- [ ] Power-of-two quantization
- [ ] Delta quantization
- [ ] Delta modulation (DPCM)
- [ ] Predictive quantization (IMA-ADPCM)

- [ ] Analog-inspired log quantization
- [ ] FPGA/ASIC-friendly quantization constraints

## Milestone 10: Decomposition + Quantization

Goal: hybrid compression strategies.

- [ ] Quantized SVD
- [ ] Quantized tensor decompositions
- [ ] Quantized LoRA
- [ ] Low-rank approximation + quantization pipelines

## Milestone 11: Calibration & Error Correction

Goal: reduce post-quantization error.

- [ ] Bias correction
- [ ] Channel equalization
- [ ] Ghost clipping

- [ ] Progressive calibration
- [ ] Scale smoothing
- [ ] Round-aware loss shaping
