import torch
from .base import Observer


class MSEObserver(Observer):
    def __init__(self, bits, num_candidates=100):
        super().__init__()
        self.bits = bits
        self.num_candidates = num_candidates
        self.min_val = None
        self.max_val = None

    def observe(self, x):
        x = x.detach().flatten()

        qmax = 2 ** (self.bits - 1) - 1
        abs_max = x.abs().max()

        if abs_max == 0:
            self.min_val = torch.tensor(0.0, device=x.device)
            self.max_val = torch.tensor(1e-6, device=x.device)
            return

        candidate_maxes = torch.linspace(
            0.1 * abs_max, abs_max, self.num_candidates, device=x.device
        )

        best_mse = float("inf")
        best_max = abs_max

        for clip_max in candidate_maxes:
            if clip_max == 0:
                continue

            scale = clip_max / qmax

            x_q = torch.clamp(torch.round(x / scale), -qmax, qmax)
            x_dq = x_q * scale

            mse = torch.mean((x - x_dq) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_max = clip_max

        self.min_val = -best_max
        self.max_val = best_max

    def get_range(self):
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("No data observed yet.")
        return self.min_val, self.max_val
