from __future__ import annotations

import numpy as np
import torch

from blpo.returns import episode_norm_reward, step_norm_reward, to_hashable


def test_to_hashable_is_stable_for_nested_values() -> None:
    left = {"b": np.asarray([1, 2]), "a": [True, 3.0]}
    right = {"a": [True, 3.0], "b": np.asarray([1, 2])}
    assert to_hashable(left) == to_hashable(right)


def test_episode_advantage_is_relative_within_task() -> None:
    rewards = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    mask = torch.ones_like(rewards)
    result = episode_norm_reward(
        rewards,
        mask,
        np.asarray(["task", "task"], dtype=object),
        np.asarray(["t0", "t1"], dtype=object),
        remove_std=True,
    )
    assert torch.allclose(result[:, 0], torch.tensor([-1.0, 1.0]))
    assert torch.allclose(result[:, 0], result[:, 1])


def test_step_advantage_broadcasts_to_response_tokens() -> None:
    rewards = torch.tensor([1.0, 3.0, 9.0])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    groups = np.asarray(["same", "same", "single"], dtype=object)
    result = step_norm_reward(rewards, mask, groups, remove_std=True)
    assert torch.allclose(result[0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(result[1], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result[2], torch.zeros(2))
