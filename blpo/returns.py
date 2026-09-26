"""Trajectory and grouped-return utilities used by BLPO.

The functions in this module are framework agnostic: ``batch`` only needs the
``batch`` and ``non_tensor_batch`` mappings used by the training adapter.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch


def to_hashable(value: Any) -> Any:
    """Recursively convert common array/container values to stable keys."""

    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(to_hashable(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), to_hashable(item)) for key, item in value.items()))
    raise TypeError(f"unsupported key type: {type(value)!r}")


def compute_step_discounted_returns(batch: Any, gamma: float) -> torch.Tensor:
    """Return the discounted environment return at every decision step."""

    rewards = np.asarray(batch.non_tensor_batch["rewards"], dtype=np.float32)
    trajectory_ids = np.asarray(batch.non_tensor_batch["traj_uid"])
    returns = np.zeros_like(rewards, dtype=np.float32)

    for trajectory_id in np.unique(trajectory_ids):
        rows = np.flatnonzero(trajectory_ids == trajectory_id)
        running = 0.0
        for row in rows[::-1]:
            running = float(rewards[row]) + float(gamma) * running
            returns[row] = running

    device = batch.batch["input_ids"].device
    return torch.as_tensor(returns, dtype=torch.float32, device=device)


def _group_moments(
    scores: torch.Tensor,
    keys: Iterable[Any],
    *,
    epsilon: float,
    remove_std: bool,
) -> torch.Tensor:
    grouped: dict[Any, list[int]] = defaultdict(list)
    for row, key in enumerate(keys):
        grouped[to_hashable(key)].append(row)

    normalized = scores.clone().float()
    for rows in grouped.values():
        values = scores[rows].float()
        mean = values.mean()
        if len(rows) == 1 or remove_std:
            normalized[rows] = values - mean
        else:
            std = values.std(unbiased=True)
            normalized[rows] = (values - mean) / (std + epsilon)
    return normalized


def episode_norm_reward(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    compute_mean_std_cross_steps: bool = True,
) -> torch.Tensor:
    """Broadcast trajectory-relative advantage over each response token."""

    scores = token_level_rewards.sum(dim=-1)
    if compute_mean_std_cross_steps:
        keys = index
    else:
        # One score per trajectory prevents long trajectories from being
        # overrepresented when the caller requests trajectory-only moments.
        first_rows: dict[tuple[Any, Any], int] = {}
        for row, pair in enumerate(zip(index, traj_index)):
            first_rows.setdefault((to_hashable(pair[0]), to_hashable(pair[1])), row)
        trajectory_scores = torch.stack([scores[row] for row in first_rows.values()])
        trajectory_groups = [pair[0] for pair in first_rows]
        group_values = _group_moments(
            trajectory_scores,
            trajectory_groups,
            epsilon=epsilon,
            remove_std=remove_std,
        )
        value_by_pair = dict(zip(first_rows, group_values))
        normalized = torch.stack(
            [value_by_pair[(to_hashable(group), to_hashable(traj))] for group, traj in zip(index, traj_index)]
        )
        return normalized.unsqueeze(-1) * response_mask

    normalized = _group_moments(
        scores,
        keys,
        epsilon=epsilon,
        remove_std=remove_std,
    )
    return normalized.unsqueeze(-1) * response_mask


def step_norm_reward(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
) -> torch.Tensor:
    """Normalize scalar decision returns inside a local comparison group."""

    normalized = _group_moments(
        step_rewards,
        index,
        epsilon=epsilon,
        remove_std=remove_std,
    )
    return normalized.unsqueeze(-1) * response_mask


def build_reference_groups(observations: np.ndarray, task_ids: np.ndarray) -> np.ndarray:
    """Build deterministic task-local exact-state groups for audit checks."""

    result = np.empty(len(observations), dtype=object)
    group_ids: dict[tuple[Any, Any], int] = {}
    for row, (observation, task_id) in enumerate(zip(observations, task_ids)):
        key = (to_hashable(task_id), to_hashable(observation))
        group_ids.setdefault(key, len(group_ids))
        result[row] = group_ids[key]
    return result


def compute_reference_group_advantage(
    *,
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference exact-state computation used only for alignment assertions."""

    remove_std = mode == "mean_norm"
    episode = episode_norm_reward(
        token_level_rewards,
        response_mask,
        index,
        traj_index,
        epsilon,
        remove_std,
    )
    groups = build_reference_groups(anchor_obs, index)
    local = step_norm_reward(step_rewards, response_mask, groups, epsilon, remove_std)
    scores = episode + step_advantage_w * local
    return scores, scores
