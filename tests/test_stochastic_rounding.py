import torch

from inwhale.rounding.stochastic import StochasticRounding


def test_seed_reproducibility():
    x = torch.tensor([2.3, 5.7, 1.2, -1.8, -3.1], dtype=torch.float32)

    r1 = StochasticRounding(seed=65)
    r2 = StochasticRounding(seed=65)

    y1 = r1.round(x)
    y2 = r2.round(x)

    assert torch.allclose(y1, y2)


def test_output_is_floor_or_ceil():
    x = torch.tensor([2.3, 5.7, 1.2, -1.8, -3.1, 0.0, 10.9999], dtype=torch.float32)
    r = StochasticRounding(seed=123)

    y = r.round(x)
    floor_x = torch.floor(x)

    assert torch.all((y == floor_x) | (y == floor_x + 1))


def test_fraction_probability_matches_approximately():
    # Use multiple fixed seeds to obtain reproducible sampling from the PRNG
    x = torch.tensor([2.3, 5.7, 1.2, -1.8, -3.1, 4.5], dtype=torch.float32)
    floor_x = torch.floor(x)
    frac = x - floor_x

    seeds = list(range(1000))  
    # 1000 independent samples per element

    ups = torch.zeros_like(x)
    for s in seeds:
        r = StochasticRounding(seed=s)
        y = r.round(x)
        ups += (y > floor_x).float()

    prop_up = ups / float(len(seeds))

    # With 1000 samples, 3-sigma error ~ 0.047 for worst-case p=0.5
    assert torch.allclose(prop_up, frac, atol=0.06, rtol=0.0)
