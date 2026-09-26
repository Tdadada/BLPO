"""Minimal stateful training adapter for BLPO.

The policy optimizer and distributed rollout engine intentionally remain
external.  This adapter owns all BLPO-specific state, computes the advantage
tensor consumed by PPO/GRPO-style actor updates, and checkpoints the memory.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import torch

from ..algorithm import BLPOConfig, BLPOMemory, compute_blpo_advantage
from ..returns import compute_step_discounted_returns


class BLPOEstimator:
    """Stateful BLPO estimator that can be embedded in an RL trainer."""

    MEMORY_FILENAME = "blpo_memory.json"

    def __init__(self, config: BLPOConfig):
        self.config = config
        self.memory = BLPOMemory(half_life=config.ema_half_life)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BLPOEstimator":
        """Construct from the ``blpo`` section of a YAML/Hydra config."""

        allowed = {item.name for item in fields(BLPOConfig)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown BLPO configuration fields: {sorted(unknown)}")
        return cls(BLPOConfig(**dict(values)))

    def compute(
        self,
        *,
        batch: Any,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        update: int,
        gamma: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], tuple[dict[str, Any], ...]]:
        """Compute BLPO advantages for one rollout update.

        The batch must provide the environment metadata consumed by
        :func:`blpo.keys.build_key_batch`; see ``docs/integration.md``.
        """

        step_returns = compute_step_discounted_returns(batch, gamma=gamma)
        return compute_blpo_advantage(
            batch=batch,
            token_level_rewards=token_level_rewards,
            step_rewards=step_returns,
            response_mask=response_mask,
            table=self.memory,
            config=self.config,
            update=update,
        )

    def save(self, checkpoint_dir: str | Path) -> Path:
        """Atomically save BLPO memory next to a policy checkpoint."""

        path = Path(checkpoint_dir) / self.MEMORY_FILENAME
        self.memory.save(path)
        return path

    def load(self, checkpoint_dir: str | Path) -> Path:
        """Restore BLPO memory when resuming policy training."""

        path = Path(checkpoint_dir) / self.MEMORY_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"missing BLPO memory checkpoint: {path}")
        self.memory.load(path)
        return path
