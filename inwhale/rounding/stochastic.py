import torch

from .base import RoundingStrategy


class StochasticRounding(RoundingStrategy):
    """
    when we create StochasticRounding(65) (65 is a random seed i chose), pytorch created a Generator object.
    The generator maintains an internal state (a sequence of numbers based on a pseudo random number generator algorithm (Mersenne Twister)).

    manual_seed(65) initializes this state to a specific starting point. 
    This state is like a deterministic "random number tape" that will produce the same sequence every time we start from the seed 65.

    The rounding algorithm is better explained with an example:
    1. we get the floored integer parts:
    floor_x = torch.floor(x)
    say floor_x = [2.0, 5.0, 1.0]

    then, we extract the fractional part:
    frac = x - floor_x
    frac = [0.3, 0.7, 0.2] ->> these represrnt the probability of rounding up

    NOW we generate random values (THE KEY STEP):
    random_vals = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=self.generator)

    a. the generator's internal state (initialised by seed 65), produces a deterministic sequence:
    seed_65 -> 
    state_0 : produces 0.456
    state_1 : produces 0.123
    state_2 : produces 0.789 (and so on...)

    b. each call to the generator:
    takes the current state, applies some mathematical transformation (PRNG algorithm), produces a flat float in [0, 1) and updates the state for the next call.

    c. same seed, same sequence. running with same seed=65 will give indentical random values.

    Let's see how probabilistic decision takes place now:

    random_vals < frac

    we compare element wise:
    [0.456, 0.123, 0.891] < [0.3, 0.7, 0.2]
    [False,  True, False] # boolean mask
    [0.0,    1.0,   0.0] # after .float()

    final result:
    floor_x + mask = [2.0, 5.0, 1.0] + [0.0, 1.0, 0.0] = [2.0, 6.0, 1.0]


    NOTE: why this is stochastic yet deterministic?

    a. stochastic:
        > 2.3 has 30% chance to round to 3, 70% to round to 2
        > 5.7 has 70% chance to round to 6, 30% to round to 5

    b. deterministic 
        > same seed -> same random seq -> same roundin decisions
        > without seed -> uses global random state -> non reproducible
    """
    
    def __init__(self, seed=None):
        self.seed = seed
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def round(self, x):
        floor_x = torch.floor(x)
        frac = x - floor_x
        if self.generator is not None:
            random_vals = torch.rand(
                x.shape, dtype=x.dtype, device=x.device, generator=self.generator
            )
        else:
            # fallback uses global RNG state 
            # kept without rand_like for compatibility
            random_vals = torch.rand(x.shape, dtype=x.dtype, device=x.device)
        return floor_x + (random_vals < frac).float()
