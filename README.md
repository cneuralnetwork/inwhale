

# inwhale

![alt text](assets/logo.jpg)


An educational library for understanding, exploring, and implementing **all major quantization techniques** used in machine learning, signal processing, LLM compression, and efficient inference.

inwhale aims to be the most complete learning-oriented quantization library available — covering everything from uniform quantization to GPTQ, LSQ, delta modulation, product quantization, and dozens of advanced methods.

---


# Features

inwhale supports (or plans to support) a comprehensive list of quantization methods, organized into categories. These represent virtually *all* quantization techniques seen in ML, DSP, and LLM research.

---

# Supported Quantization Methods

## 1. Uniform Quantization

* Symmetric uniform
* Asymmetric uniform
* Affine quantization
* Mid-rise / mid-tread quantizers
* Dead-zone quantizers
* Per-tensor and per-channel quantization
* Mixed-precision schemes
* Sub-byte (4-bit, 2-bit, 1-bit) quantization

## 2. Non-Uniform Quantization

* Logarithmic quantization (power-of-two)
* µ-law companding
* A-law companding
* Learned non-uniform (LUT-based) quantizers
* Vector quantizers
* Lattice quantizers
* Residual quantization
* Additive quantization
* Product quantization (PQ, OPQ, RQ)
* VQ-VAE–style codebook quantization

## 3. Rounding Strategies

* Nearest rounding
* Stochastic rounding
* Banker’s rounding
* Round-away-from-zero
* Floor / ceil
* Truncation
* Adaptive rounding (AdaRound)
* Gradient-learned rounding (LSQ+)

## 4. Observers / Range Estimators

* MinMax observer
* Moving average observer
* Percentile observer
* MSE-based scale search
* KL-divergence observer
* Histogram observers
* Per-channel observers
* Gradient-based scaling (LSQ family)

## 5. Post-Training Quantization (PTQ)

* Static PTQ
* Dynamic PTQ
* Weight-only quantization
* SmoothQuant
* GPTQ (second-order PTQ)
* AWQ (outlier-aware PTQ)
* ZeroQuant
* RQ / Relaxed Quantization
* Outlier splitting (LLM.int8)
* Blockwise quantization

## 6. Quantization-Aware Training (QAT)

* Fake quantization
* LSQ (Learned Step Size Quantization)
* LSQ+
* INQ (Incremental Quantization)
* DoReFa-Net
* Ternary quantization (TTQ)
* Binary networks (XNOR-Net, BinaryConnect, ABC-Net)
* QAT for transformers and attention modules

## 7. LLM-Specific Quantization

* NF4 / FP4 formats
* Group-wise quantization
* Per-row / per-column quantization
* Attention-specific quantization
* Outlier-aware activation quantization
* Quantized LoRA (QLoRA)
* Mixed precision activations (8-bit + 16-bit outliers)

## 8. Vector & Codebook Quantization

* PQ / OPQ / RQ / AQ
* Neural codebook learning
* Multi-level vector quantization

## 9. Hardware-Inspired Quantization

* Power-of-two quantization
* Delta quantization
* Delta modulation (DPCM)
* Predictive quantization (IMA-ADPCM)
* Analog-inspired log quantization
* Quantization for FPGA/ASIC constraints

## 10. Decomposition + Quantization

* Quantized SVD
* Quantized tensor decompositions
* Quantized LoRA
* Low-rank approximations + quantization combos

## 11. Calibration & Error Correction

* Bias correction
* Channel equalization
* Ghost clipping
* Progressive calibration
* Scale smoothing
* Round-aware loss shaping

This makes inwhale suitable not just for basic quantization, but for research, teaching, and experimentation with cutting-edge LLM compression techniques.

---

# Installation

```
git clone https://github.com/<your-username>/inwhale.git
cd inwhale
pip install -r requirements.txt
```

---

# Quick Start

```python
import torch
from inwhale.core.uniform import UniformQuantizer

x = torch.tensor([1.2, -0.7, 0.4])
quant = UniformQuantizer(bits=8)

qx = quant.quantize(x)
dx = quant.dequantize(qx)

print(dx)
```

---

# Project Structure

```
inwhale/
│
├── inwhale/
│   ├── core/
│   │   ├── uniform.py
│   │   ├── affine.py
│   │   ├── rounding.py
│   │   ├── observers.py
│   │   └── quantizer.py
│   │
│   ├── ptq/
│   │   ├── linear_ptq.py
│   │   ├── conv_ptq.py
│   │   ├── gptq.py
│   │   ├── awq.py
│   │   └── smoothquant.py
│   │
│   ├── qat/
│   │   ├── fake_quant.py
│   │   ├── lsq.py
│   │   ├── binary.py
│   │   └── qmodules.py
│   │
│   ├── llm/
│   │   ├── nf4.py
│   │   ├── fp4.py
│   │   ├── gptq_blocks.py
│   │   └── group_quant.py
│   │
│   ├── vector/
│   │   ├── pq.py
│   │   ├── opq.py
│   │   ├── rq.py
│   │   └── aq.py
│   │
│   ├── utils/
│   │   ├── calibration.py
│   │   ├── error_metrics.py
│   │   └── plotting.py
│   │
│   └── datasets/
│       └── calibration_sets.py
│
├── examples/
├── tests/
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```

---

# Goals of the Project

* Provide the most **complete educational coverage** of quantization techniques
* Enable students to learn quantization mathematically and practically
* Make research papers easier to understand through clean code
* Support LLM compression experiments
* Build a collaborative quantization learning hub for your club


---

# Contributing

See **CONTRIBUTING.md** for workflows, coding rules, and guidelines.

Beginner-friendly issues are labeled and designed to teach real quantization concepts.

---

# License

MIT License.

