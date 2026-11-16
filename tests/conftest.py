import numpy as np
import pandas as pd
import pytest

from trendcarry import synthetic


@pytest.fixture(scope="session")
def sim():
    return synthetic.generate(seed=11, small=True, end="2009-12-31")


@pytest.fixture(scope="session")
def ds(sim):
    return synthetic.to_dataset(sim)


@pytest.fixture(scope="session")
def null_ds():
    """Pure noise: no trend and no carry premium."""
    return synthetic.to_dataset(synthetic.generate(seed=3, small=True, trend_strength=0.0, carry_premium=0.0))


def toy_prices(n=400, codes=("A", "B"), seed=0, start="2000-01-03"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.standard_normal((n, len(codes))) * 0.01, axis=0)),
                        index=idx, columns=list(codes))
