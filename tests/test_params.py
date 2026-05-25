"""Tests for param budget."""

from sam.core.sam_core import SamCore
from sam.config import MAX_TOTAL_PARAMS


def test_core_within_3m_params():
    core = SamCore(latent_dim=128)
    n = core.total_params()
    assert n < MAX_TOTAL_PARAMS
    assert n > 100_000
