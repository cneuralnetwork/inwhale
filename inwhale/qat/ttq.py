import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from inwhale.core.quantizer import BaseQuantizer


def compute_ttq_threshold(weight: torch.Tensor, threshold_ratio: float = 0.05) -> torch.Tensor:
    """
    Compute the quantization threshold Delta for Trained Ternary Quantization (TTQ).

    Following Zhu et al. (ICLR 2017), the threshold Delta is set proportional
    to the maximum absolute value in the weight tensor:
        Delta = threshold_ratio * max(|weight|)

    Args:
        weight: Input weight tensor (floating-point).
        threshold_ratio: Fraction of max absolute weight used as threshold.
            Defaults to 0.05 per Zhu et al. (2017).

    Returns:
        Scalar tensor Delta >= 0.
    """
    if not (0.0 < threshold_ratio < 1.0):
        raise ValueError(f"threshold_ratio must be strictly between 0 and 1, got {threshold_ratio}.")
    return threshold_ratio * weight.abs().max()


def compute_ttq_scales(
    weight: torch.Tensor, delta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute initial positive (wp) and negative (wn) scale factors for TTQ.

    Following Zhu et al. (ICLR 2017), initial scaling factors are the expected
    magnitudes of the non-zero partitioned weights:
        wp = mean(weight[weight > Delta])
        wn = mean(|weight[weight < -Delta]|)

    If a partition contains no elements, the scale defaults to 1.0 (or tensor max).

    Args:
        weight: Input weight tensor.
        delta: Threshold Delta.

    Returns:
        Tuple of (wp, wn) as scalar tensors.
    """
    pos_mask = weight > delta
    neg_mask = weight < -delta

    if pos_mask.any():
        wp = weight[pos_mask].mean()
    else:
        wp = torch.tensor(1.0, dtype=weight.dtype, device=weight.device)

    if neg_mask.any():
        wn = weight[neg_mask].abs().mean()
    else:
        wn = torch.tensor(1.0, dtype=weight.dtype, device=weight.device)

    return wp, wn


class TTQFunction(torch.autograd.Function):
    """
    Custom autograd Function for Trained Ternary Quantization (TTQ).

    Forward pass:
        Quantizes full-precision weight W into ternary values {-wn, 0, +wp}:
            W^t = +wp   if W > Delta
            W^t =  0    if |W| <= Delta
            W^t = -wn   if W < -Delta
        where Delta = threshold_ratio * max(|W|).

    Backward pass:
        Computes exact analytic gradients for positive scale factor wp and
        negative scale factor wn, and applies a Straight-Through Estimator (STE)
        to propagate gradients back to full-precision weights W:

        dL / dwp = sum_{i in I_p} (dL / dW_i^t)
        dL / dwn = sum_{i in I_n} -(dL / dW_i^t)

        For full-precision weights (Zhu et al., Eq. 6):
            dL / dW_i = (dL / dW_i^t) * wp    if W_i > Delta
            dL / dW_i = (dL / dW_i^t) * 1.0   if |W_i| <= Delta
            dL / dW_i = (dL / dW_i^t) * wn    if W_i < -Delta

        This allows pruned (zeroed) weights to recover and transition into active
        weights while scaling active gradients by their respective learned scales.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        wp: torch.Tensor,
        wn: torch.Tensor,
        threshold_ratio: float = 0.05,
        gradient_mode: str = "ttq",
    ) -> torch.Tensor:
        delta = compute_ttq_threshold(weight, threshold_ratio)

        mask_p = weight > delta
        mask_n = weight < -delta
        mask_z = ~(mask_p | mask_n)

        quantized = (
            wp * mask_p.to(weight.dtype) - wn * mask_n.to(weight.dtype)
        )

        ctx.save_for_backward(weight, mask_p, mask_n, mask_z, wp, wn)
        ctx.gradient_mode = gradient_mode
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, mask_p, mask_n, mask_z, wp, wn = ctx.saved_tensors

        # Gradients for trainable positive and negative scale factors
        grad_wp = (grad_output * mask_p.to(grad_output.dtype)).sum().reshape(wp.shape)
        grad_wn = (-grad_output * mask_n.to(grad_output.dtype)).sum().reshape(wn.shape)

        # Gradients for full-precision weights via STE
        if ctx.gradient_mode == "ttq":
            ste_multiplier = (
                wp * mask_p.to(grad_output.dtype)
                + wn * mask_n.to(grad_output.dtype)
                + 1.0 * mask_z.to(grad_output.dtype)
            )
            grad_weight = grad_output * ste_multiplier
        elif ctx.gradient_mode == "ste":
            grad_weight = grad_output.clone()
        else:
            raise ValueError(f"Unknown gradient_mode: {ctx.gradient_mode}. Must be 'ttq' or 'ste'.")

        return grad_weight, grad_wp, grad_wn, None, None


def ttq_quantize(
    weight: torch.Tensor,
    threshold_ratio: float = 0.05,
    wp: Optional[Union[float, torch.Tensor]] = None,
    wn: Optional[Union[float, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize weight tensor into ternary values using Trained Ternary Quantization (TTQ).

    Functional helper that computes ternary quantized weights, discrete codes {-1, 0, +1},
    and positive/negative scale factors.

    Args:
        weight: Input weight tensor.
        threshold_ratio: Fraction of max absolute weight for threshold Delta (default 0.05).
        wp: Optional predetermined positive scale factor. If None, initialized from weight.
        wn: Optional predetermined negative scale factor. If None, initialized from weight.

    Returns:
        Tuple of:
            - quantized_weight: Float tensor with values {-wn, 0, +wp}.
            - ternary_codes: Int8 tensor with values {-1, 0, +1}.
            - wp: Positive scale factor.
            - wn: Negative scale factor.
    """
    delta = compute_ttq_threshold(weight, threshold_ratio)

    if wp is None or wn is None:
        init_wp, init_wn = compute_ttq_scales(weight, delta)
        wp = init_wp if wp is None else torch.as_tensor(wp, dtype=weight.dtype, device=weight.device)
        wn = init_wn if wn is None else torch.as_tensor(wn, dtype=weight.dtype, device=weight.device)
    else:
        wp = torch.as_tensor(wp, dtype=weight.dtype, device=weight.device)
        wn = torch.as_tensor(wn, dtype=weight.dtype, device=weight.device)

    mask_p = weight > delta
    mask_n = weight < -delta

    codes = torch.zeros_like(weight, dtype=torch.int8)
    codes[mask_p] = 1
    codes[mask_n] = -1

    quantized = wp * mask_p.to(weight.dtype) - wn * mask_n.to(weight.dtype)
    return quantized, codes, wp, wn


class TTQQuantizer(BaseQuantizer):
    """
    Trained Ternary Quantizer (TTQ).

    Category 6: Quantization-Aware Training (QAT).

    Reference:
        Zhu, Chenzhuo, Song Han, Huizi Mao, and William J. Dally.
        "Trained Ternary Quantization."
        International Conference on Learning Representations (ICLR), 2017.
        https://arxiv.org/abs/1612.01064

    Motivation:
        Standard ternary quantization (e.g. Ternary Weight Networks, TWN) restricts
        weights to {-w, 0, +w} with a single symmetric scaling factor w.
        However, deep neural networks (especially those trained with ReLU or asymmetrical
        activations) exhibit pronounced asymmetry between positive and negative weight
        distributions. Constraining positive and negative weights to have identical magnitude
        results in substantial representation loss.

        TTQ solves this by introducing two separate, learnable, full-precision parameters:
            - wp >= 0 (positive scale factor)
            - wn >= 0 (negative scale factor)

        Each weight W_i is partitioned into one of three states based on threshold Delta:
            W_i^t = +wp   if W_i > Delta
            W_i^t =  0    if |W_i| <= Delta
            W_i^t = -wn   if W_i < -Delta
        where Delta = threshold_ratio * max(|W|).

    Ternary Integer Encoding:
        Because ternary values require only 3 discrete states, each weight is represented
        in 2 bits:
            +1 -> positive state (rescaled by +wp)
             0 -> zero / pruned state
            -1 -> negative state (rescaled by -wn)
        yielding up to 16x compression compared to FP32 weights and eliminating FP32
        multiplications during inference (reducing them to additions and subtractions).

    Args:
        threshold_ratio: Fraction of max absolute weight used to compute threshold Delta.
            Default: 0.05 (Zhu et al., 2017).
        wp: Optional initial positive scale factor.
        wn: Optional initial negative scale factor.
        gradient_mode: Straight-Through Estimator mode ('ttq' or 'ste'). Default: 'ttq'.
    """

    def __init__(
        self,
        threshold_ratio: float = 0.05,
        wp: Optional[Union[float, torch.Tensor]] = None,
        wn: Optional[Union[float, torch.Tensor]] = None,
        gradient_mode: str = "ttq",
    ):
        if not (0.0 < threshold_ratio < 1.0):
            raise ValueError(f"threshold_ratio must be strictly between 0 and 1, got {threshold_ratio}.")
        if gradient_mode not in ("ttq", "ste"):
            raise ValueError(f"gradient_mode must be 'ttq' or 'ste', got {gradient_mode}.")

        super().__init__(bits=2)
        self.threshold_ratio = float(threshold_ratio)
        self.gradient_mode = gradient_mode

        # Ternary integer grid boundaries
        self.qmin = -1
        self.qmax = 1

        self.wp = None if wp is None else torch.as_tensor(wp, dtype=torch.float32)
        self.wn = None if wn is None else torch.as_tensor(wn, dtype=torch.float32)
        self.delta = None

    def _compute_parameters(self, x: torch.Tensor):
        self.delta = compute_ttq_threshold(x, self.threshold_ratio)
        init_wp, init_wn = compute_ttq_scales(x, self.delta)
        if self.wp is None:
            self.wp = init_wp
        if self.wn is None:
            self.wn = init_wn

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor into discrete ternary codes {-1, 0, +1}.

        Args:
            x: Input continuous tensor.

        Returns:
            Tensor of dtype torch.int8 with values strictly in {-1, 0, 1}.
        """
        self._compute_parameters(x)
        mask_p = x > self.delta
        mask_n = x < -self.delta

        q = torch.zeros_like(x, dtype=torch.int8)
        q[mask_p] = 1
        q[mask_n] = -1
        return q

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        """
        Dequantize discrete ternary codes {-1, 0, +1} back to continuous values {-wn, 0, +wp}.

        Args:
            q: Discrete ternary tensor with values in {-1, 0, 1}.

        Returns:
            Continuous reconstructed float tensor.
        """
        if self.wp is None or self.wn is None:
            raise RuntimeError("Quantizer must quantize a tensor before dequantization, or wp and wn must be supplied.")

        out = torch.zeros_like(q, dtype=torch.float32)
        out[q == 1] = float(self.wp)
        out[q == -1] = -float(self.wn)
        return out

    def fake_quantize(
        self,
        x: torch.Tensor,
        wp: Optional[torch.Tensor] = None,
        wn: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform differentiable fake-quantization using TTQFunction for QAT training.

        Args:
            x: Input weight tensor requiring gradients.
            wp: Trainable positive scale factor parameter. If None, uses self.wp.
            wn: Trainable negative scale factor parameter. If None, uses self.wn.

        Returns:
            Quantized tensor with autograd STE graph attached.
        """
        if wp is None or wn is None:
            self._compute_parameters(x)
            eff_wp = self.wp if wp is None else wp
            eff_wn = self.wn if wn is None else wn
        else:
            eff_wp = wp
            eff_wn = wn

        return TTQFunction.apply(
            x, eff_wp, eff_wn, self.threshold_ratio, self.gradient_mode
        )


class TTQLinear(nn.Module):
    """
    Linear layer with Trained Ternary Quantization (TTQ) for Quantization-Aware Training.

    Replaces standard nn.Linear. Weights are quantized during the forward pass using
    TTQ with learnable positive (wp) and negative (wn) scale factors. Full-precision
    weights are updated during backpropagation via Straight-Through Estimation.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.
        threshold_ratio: Fraction of max absolute weight for threshold Delta (default 0.05).
        gradient_mode: STE mode ('ttq' or 'ste'). Default: 'ttq'.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold_ratio: float = 0.05,
        gradient_mode: str = "ttq",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_ratio = float(threshold_ratio)
        self.gradient_mode = gradient_mode

        # Full-precision master weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize learnable positive and negative scale factors
        with torch.no_grad():
            delta = compute_ttq_threshold(self.weight, self.threshold_ratio)
            init_wp, init_wn = compute_ttq_scales(self.weight, delta)

        self.wp = nn.Parameter(init_wp.clone().detach())
        self.wn = nn.Parameter(init_wn.clone().detach())

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def clamp_scales(self, min_val: float = 1e-5):
        """Ensure scaling parameters remain non-negative during training."""
        with torch.no_grad():
            self.wp.clamp_(min=min_val)
            self.wn.clamp_(min=min_val)

    def get_quantized_weight(self) -> torch.Tensor:
        """Return the current ternary quantized weights."""
        return TTQFunction.apply(
            self.weight, self.wp, self.wn, self.threshold_ratio, self.gradient_mode
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_weight = self.get_quantized_weight()
        return F.linear(input, q_weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, threshold_ratio={self.threshold_ratio}, "
            f"gradient_mode='{self.gradient_mode}'"
        )


class TTQConv2d(nn.Module):
    """
    2D Convolutional layer with Trained Ternary Quantization (TTQ) for QAT.

    Replaces standard nn.Conv2d. Weights are quantized during the forward pass into
    ternary values {-wn, 0, +wp}.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides of the input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections from input to output channels. Default: 1.
        bias: If True, adds a learnable bias. Default: True.
        threshold_ratio: Fraction of max absolute weight for threshold Delta (default 0.05).
        gradient_mode: STE mode ('ttq' or 'ste'). Default: 'ttq'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        threshold_ratio: float = 0.05,
        gradient_mode: str = "ttq",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.threshold_ratio = float(threshold_ratio)
        self.gradient_mode = gradient_mode

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            delta = compute_ttq_threshold(self.weight, self.threshold_ratio)
            init_wp, init_wn = compute_ttq_scales(self.weight, delta)

        self.wp = nn.Parameter(init_wp.clone().detach())
        self.wn = nn.Parameter(init_wn.clone().detach())

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def clamp_scales(self, min_val: float = 1e-5):
        """Ensure scaling parameters remain non-negative during training."""
        with torch.no_grad():
            self.wp.clamp_(min=min_val)
            self.wn.clamp_(min=min_val)

    def get_quantized_weight(self) -> torch.Tensor:
        """Return the current ternary quantized weights."""
        return TTQFunction.apply(
            self.weight, self.wp, self.wn, self.threshold_ratio, self.gradient_mode
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_weight = self.get_quantized_weight()
        return F.conv2d(
            input,
            q_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, bias={self.bias is not None}, "
            f"threshold_ratio={self.threshold_ratio}, gradient_mode='{self.gradient_mode}'"
        )
