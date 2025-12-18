import torch

from .base import Observer


class PercentileObserver(Observer):
    def __init__(self, lower_quantile=0.001, upper_quantile=0.999):
        super().__init__()

        if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
            raise ValueError("Quantiles must satisfy 0.0 <= lower < upper <= 1.0")

        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

        self.min_val = None
        self.max_val = None

    def observe(self, x):
        if x.numel() == 0:
            raise ValueError("Cannot observe empty tensor")
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values")

        x = x.detach()
        x_flat = x.flatten()

        # min_x = torch.quantile(x_flat, self.lower_quantile)
        # max_x = torch.quantile(x_flat, self.upper_quantile)

        quantiles = torch.quantile(x_flat, torch.tensor([self.lower_quantile, self.upper_quantile], device=x.device))
        min_x, max_x = quantiles[0], quantiles[1]

        self.min_val = min_x
        self.max_val = max_x

    def get_range(self):
        if self.min_val is None:
            raise RuntimeError("No data observed yet.")
        return self.min_val, self.max_val
