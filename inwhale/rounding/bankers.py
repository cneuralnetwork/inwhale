import torch

from .base import RoundingStrategy


class BankersRounding(RoundingStrategy):
    """
    Docstring for BankersRounding

    we take a number, and floor it to get the integer part.
    then we get the fractional part.
    we check whether the fractional part is exactly 0.5.

    where it is exactly 0.5, we check whether the floored integer part is even or odd.
    if even, we keep it as is (the floored value).
    if odd, we add 1 to the floored value.
    otherwise, we just round normally.
    """
    def round(self, x):
        floored = torch.floor(x)
        fractional = x - floored
        is_half = fractional == 0.5

        rounded = torch.where(
            is_half,
            torch.where((floored % 2) == 0, floored, floored + 1),
            torch.round(x),
        )
        return rounded
