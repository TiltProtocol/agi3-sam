"""Dynamic action masking tests."""

import torch

from sam.core.policy import ActionHead


def test_mask_excludes_unavailable():
    head = ActionHead(latent_dim=32, num_actions=8)
    z = torch.zeros(32)
    logits = head.masked_logits(z, [1, 3])
    assert logits[1] > logits[0]
    assert logits[3] > logits[0]
    assert torch.isinf(logits[2]) and logits[2] < 0


def test_select_only_available():
    head = ActionHead(latent_dim=32, num_actions=8)
    z = torch.randn(32)
    for _ in range(20):
        aid = head.select_action(z, [4], temperature=0.0)
        assert aid == 4
