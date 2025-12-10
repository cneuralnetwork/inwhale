

# Contributing

Thank you for your interest in contributing to this project. The goal of this library is to provide clear, readable implementations of quantization techniques for learning and experimentation. Contributions of all levels are welcome.

Before starting work on anything, read this document carefully.

**PLEASE CONTRIBUTE SEQUENTIALLY FROM ISSUE 1 AND HENCEFORTH**

---

## How to Contribute

### 1. Fork the Repository

* Press **Fork** on GitHub.
* Clone your fork:

```
git clone https://github.com/<your-username>/inwhale.git
cd inwhale
```

### 2. Create a New Branch

Use descriptive names:

```
git checkout -b feature-uniform-quantizer
```

Examples:

* `fix-rounding-bug`
* `add-minmax-observer`
* `docs-qat-overview`

Never commit directly to `main`.

---

## Development Setup

Install dependencies:

```
pip install -r requirements.txt
```

Run tests:

```
pytest
```

All new features must include test cases under `tests/`.

---

## Code Style

* Keep functions short and readable.
* Add docstrings explaining:

  * the purpose
  * the math
  * expected input and output
* Prefer clarity over optimization. This library is educational.
* Use type hints where possible.
* Follow the existing directory structure:

```
inwhale/core/...
inwhale/ptq/...
inwhale/qat/...
```

---

## Adding New Quantization Components

When implementing a new quantizer, observer, rounding method, or QAT module:

1. Place the implementation in the correct folder.
2. Add a brief explanation of the idea in comments/docstrings.
3. Add a simple test covering correctness.
4. Add an example (if meaningful) in the `examples/` folder or as a short code snippet in the docstring.

### Checklist Before Submitting

* [ ] Code is placed in the appropriate module
* [ ] Docstrings included
* [ ] Tests added
* [ ] Code is formatted and readable
* [ ] No unused print statements or debug code

---

## Commit Guidelines

Use descriptive messages:

* `add stochastic rounding`
* `fix affine quant scale computation`
* `refactor minmax observer for clarity`

---

## Pull Request Process

1. Push your branch to your fork:

```
git push origin feature-your-feature
```

2. Open a Pull Request.
3. In the PR description, include:

   * What you implemented
   * Why it is useful
   * Any mathematical references
   * Tests added
4. Wait for review and make requested updates.

PRs should be small and focused. Avoid mixing multiple features.

---

## Issue Guidelines

If you want to work on an issue:

* Comment on the issue so it can be assigned to you.
* Ask for clarification if anything is unclear.

If you’re creating a new issue:

Include:

* Problem description
* Expected behaviour
* Steps to reproduce (if bug)
* Suggested implementation path (if feature)

We label issues as:

* **good first issue**
* **intermediate**
* **advanced**

Choose one based on your comfort level.

---

## Documentation Contributions

Improving documentation, adding explanations, or writing educational notebooks is strongly encouraged.

Examples:

* Write a tutorial demonstrating how symmetric vs asymmetric quantization differ.
* Add plots showing quantization error.
* Add explanations inside docstrings.

---


