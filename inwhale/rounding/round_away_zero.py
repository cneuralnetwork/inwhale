import torch

from .base import RoundingStrategy


class RoundAwayFromZero(RoundingStrategy):
    """
    This rounding strategy rounds values away from zero, by computing
    sign(x) * ceil(abs(x)).
    Positive values round up, negative values round down.
    """
    def round(self, x):
        """
        Docstring for round
        
        :param x: input tensor of numeric values
        :return: tensor with values rounded away from zero
        """
        return torch.sign(x) * torch.ceil(torch.abs(x))
