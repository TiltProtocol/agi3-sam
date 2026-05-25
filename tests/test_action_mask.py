"""Tests for dynamic action masking."""

import torch

from sam.core.policy import ActionHead


def test_masked_logits_only_available():
    head = ActionHead(latent_dim=16, num_actions=8)
    z = torch.randn(16)
    logits = head.masked_logits(z, available_actions=[1, 3])
    assert logits[1].item() > float("-inf")
    assert logits[3].item() > float("-inf")
    assert logits[0].item() == float("-inf")
    assert logits[2].item() == float("-inf")


def test_select_action_respects_mask():
    head = ActionHead(latent_dim=16, num_actions=8)
    z = torch.zeros(16)
    for _ in range(20):
        aid = head.select_action(z, available_actions=[2, 4], temperature=0.1)
        assert aid in (2, 4)
