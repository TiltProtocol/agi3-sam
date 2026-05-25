"""SAM core param budget tests."""

from sam.core.sam_core import SamCore
from sam.config import MAX_TOTAL_PARAMS


def test_core_within_param_cap():
    core = SamCore(latent_dim=128)
    assert core.total_params() <= MAX_TOTAL_PARAMS
    assert core.within_budget()
