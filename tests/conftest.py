import pytest
import torch

from weightkit import apply_op
from weightkit.testing import from_store, random_model, to_store


@pytest.fixture
def ids():
    torch.manual_seed(123)
    return torch.randint(0, 64, (3, 10))


def logits(store, ids):
    with torch.no_grad():
        return from_store(store)(ids)


def make(seed=0, **cfg):
    m = random_model(seed, **cfg)
    return m, to_store(m)
