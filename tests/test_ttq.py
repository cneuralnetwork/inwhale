import math
import pytest
import torch
import torch.nn as nn

from inwhale.qat.ttq import (
    TTQConv2d,
    TTQFunction,
    TTQLinear,
    TTQQuantizer,
    compute_ttq_scales,
    compute_ttq_threshold,
    ttq_quantize,
)


def test_ttq_quantizer_init():
    quantizer = TTQQuantizer(threshold_ratio=0.05)
    assert quantizer.bits == 2
    assert quantizer.qmin == -1
    assert quantizer.qmax == 1
    assert quantizer.threshold_ratio == 0.05
    assert quantizer.gradient_mode == "ttq"


def test_invalid_parameters():
    with pytest.raises(ValueError, match="threshold_ratio"):
        TTQQuantizer(threshold_ratio=0.0)

    with pytest.raises(ValueError, match="threshold_ratio"):
        TTQQuantizer(threshold_ratio=1.0)

    with pytest.raises(ValueError, match="threshold_ratio"):
        TTQQuantizer(threshold_ratio=-0.1)

    with pytest.raises(ValueError, match="gradient_mode"):
        TTQQuantizer(gradient_mode="invalid")


def test_ttq_discrete_code_grid():
    torch.manual_seed(42)
    w = torch.randn(64, 64)
    quantizer = TTQQuantizer(threshold_ratio=0.05)
    q = quantizer.quantize(w)

    assert q.dtype == torch.int8
    unique_vals = set(q.unique().tolist())
    assert unique_vals.issubset({-1, 0, 1})
    assert -1 in unique_vals
    assert 0 in unique_vals
    assert 1 in unique_vals


def test_ttq_dequantize_values():
    w = torch.tensor([-1.0, -0.5, 0.0, 0.02, 0.4, 0.8])
    quantizer = TTQQuantizer(threshold_ratio=0.05, wp=0.6, wn=0.75)
    q = quantizer.quantize(w)
    deq = quantizer.dequantize(q)

    assert deq.dtype == torch.float32
    assert torch.equal(deq == 0.6, q == 1)
    assert torch.equal(deq == -0.75, q == -1)
    assert torch.equal(deq == 0.0, q == 0)


def test_ttq_exact_threshold_boundary():
    # max magnitude is 1.0, threshold_ratio is 0.1 -> Delta = 0.1
    w = torch.tensor([-1.0, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 1.0])
    quantizer = TTQQuantizer(threshold_ratio=0.1)
    q = quantizer.quantize(w)

    expected_q = torch.tensor([-1, -1, 0, 0, 0, 0, 0, 1, 1], dtype=torch.int8)
    assert torch.equal(q, expected_q)


def test_ttq_asymmetric_scales():
    # Asymmetric distribution: large positive peaks, small negative peaks
    w = torch.tensor([-0.2, -0.1, 0.0, 0.5, 0.9])
    q, codes, wp, wn = ttq_quantize(w, threshold_ratio=0.05)

    assert wp > wn  # Positive scale should be substantially larger than negative scale
    assert q[w == 0.9].item() == pytest.approx(wp.item(), rel=1e-4)
    assert q[w == -0.2].item() == pytest.approx(-wn.item(), rel=1e-4)
    assert q[w == 0.0].item() == 0.0


def test_ttq_functional_api():
    w = torch.randn(10, 10)
    quantized, codes, wp, wn = ttq_quantize(w, threshold_ratio=0.05)

    quantizer = TTQQuantizer(threshold_ratio=0.05, wp=wp, wn=wn)
    q_class = quantizer.quantize(w)
    deq_class = quantizer.dequantize(q_class)

    assert torch.equal(codes, q_class)
    assert torch.allclose(quantized, deq_class, atol=1e-6)


def test_ttq_autograd_gradients():
    # Explicit tensor values to test exact analytic gradient derivation
    w = torch.tensor([-0.8, -0.3, 0.01, 0.03, 0.4, 0.9], requires_grad=True)
    wp = torch.tensor(0.5, requires_grad=True)
    wn = torch.tensor(0.7, requires_grad=True)

    # Forward pass: max is 0.9, delta = 0.05 * 0.9 = 0.045
    # pos: indices 4, 5 (0.4, 0.9)
    # neg: indices 0, 1 (-0.8, -0.3)
    # zero: indices 2, 3 (0.01, 0.03)
    out = TTQFunction.apply(w, wp, wn, 0.05, "ttq")

    grad_output = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out.backward(grad_output)

    # dL/dwp = sum(grad_output[pos]) = 5.0 + 6.0 = 11.0
    assert wp.grad.item() == pytest.approx(11.0, rel=1e-5)

    # dL/dwn = -sum(grad_output[neg]) = -(1.0 + 2.0) = -3.0
    assert wn.grad.item() == pytest.approx(-3.0, rel=1e-5)

    # dL/dW via TTQ STE:
    # pos: grad_output * wp = [5.0*0.5, 6.0*0.5] = [2.5, 3.0]
    # neg: grad_output * wn = [1.0*0.7, 2.0*0.7] = [0.7, 1.4]
    # zero: grad_output * 1.0 = [3.0, 4.0]
    expected_w_grad = torch.tensor([0.7, 1.4, 3.0, 4.0, 2.5, 3.0])
    assert torch.allclose(w.grad, expected_w_grad, atol=1e-5)


def test_ttq_gradient_mode_ste():
    w = torch.tensor([-0.8, 0.02, 0.9], requires_grad=True)
    wp = torch.tensor(0.5, requires_grad=True)
    wn = torch.tensor(0.7, requires_grad=True)

    out = TTQFunction.apply(w, wp, wn, 0.05, "ste")
    grad_output = torch.tensor([1.0, 2.0, 3.0])
    out.backward(grad_output)

    # In standard STE mode, gradient w.r.t W is passed through directly
    assert torch.allclose(w.grad, grad_output, atol=1e-5)


def test_ttq_linear_layer():
    torch.manual_seed(0)
    layer = TTQLinear(in_features=8, out_features=4, bias=True)

    x = torch.randn(3, 8)
    out = layer(x)

    assert out.shape == (3, 4)
    loss = out.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.wp.grad is not None
    assert layer.wn.grad is not None
    assert layer.bias.grad is not None

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    orig_wp = layer.wp.item()
    optimizer.step()
    layer.clamp_scales(min_val=1e-4)

    assert layer.wp.item() != orig_wp
    assert layer.wp.item() >= 1e-4 - 1e-7
    assert layer.wn.item() >= 1e-4 - 1e-7


def test_ttq_conv2d_layer():
    torch.manual_seed(0)
    conv = TTQConv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)

    x = torch.randn(2, 3, 16, 16)
    out = conv(x)

    assert out.shape == (2, 6, 16, 16)
    loss = out.mean()
    loss.backward()

    assert conv.weight.grad is not None
    assert conv.wp.grad is not None
    assert conv.wn.grad is not None
    assert conv.bias.grad is not None


def test_ttq_training_convergence():
    torch.manual_seed(42)
    x = torch.randn(128, 8)
    # Simple linear separation task
    y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)

    model = nn.Sequential(
        TTQLinear(8, 16),
        nn.ReLU(),
        TTQLinear(16, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCEWithLogitsLoss()

    initial_loss = None
    final_loss = None

    for epoch in range(25):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, TTQLinear):
                    m.clamp_scales(min_val=1e-4)

        if epoch == 0:
            initial_loss = loss.item()
        if epoch == 24:
            final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, f"Loss should reduce significantly: {initial_loss} -> {final_loss}"


def test_ttq_edge_cases():
    # All positive weights
    w_pos = torch.tensor([0.2, 0.5, 0.8])
    q_pos, codes_pos, wp_pos, wn_pos = ttq_quantize(w_pos)
    assert (codes_pos >= 0).all()

    # All negative weights
    w_neg = torch.tensor([-0.2, -0.5, -0.8])
    q_neg, codes_neg, wp_neg, wn_neg = ttq_quantize(w_neg)
    assert (codes_neg <= 0).all()

    # Dequantize without quantize raises RuntimeError
    uninit_q = TTQQuantizer()
    with pytest.raises(RuntimeError, match="Quantizer must quantize"):
        uninit_q.dequantize(torch.tensor([0, 1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ttq_cuda():
    device = torch.device("cuda")
    w = torch.randn(32, 32, device=device)
    layer = TTQLinear(32, 16).to(device)

    x = torch.randn(8, 32, device=device)
    out = layer(x)
    assert out.is_cuda
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad.is_cuda
    assert layer.wp.grad.is_cuda
    assert layer.wn.grad.is_cuda
