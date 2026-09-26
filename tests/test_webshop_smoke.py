from __future__ import annotations

import numpy as np
import torch

from blpo import BLPOConfig
from blpo.integration import BLPOEstimator


class Batch:
    def __init__(self) -> None:
        state = {
            "page_type": "search_results",
            "query": ["red", "mug"],
            "available_actions": {"clickables": ["b001", "b002", "next >"]},
            "selected_options": {},
            "action_counts": {},
        }
        self.batch = {"input_ids": torch.ones((4, 2), dtype=torch.long)}
        self.non_tensor_batch = {
            "uid": np.asarray(["task", "task", "task", "task"], dtype=object),
            "traj_uid": np.asarray(["t0", "t1", "t2", "t3"], dtype=object),
            "anchor_obs": np.asarray(
                ["b001 [SEP] red ceramic mug [SEP] b002 [SEP] blue bowl"] * 4,
                dtype=object,
            ),
            "text_action": np.asarray(
                ["click[b001]", "click[b001]", "click[b002]", "click[b002]"],
                dtype=object,
            ),
            "rewards": np.asarray([1.0, 0.8, 0.1, 0.0], dtype=np.float32),
            "episode_rewards": np.asarray([1.0, 0.8, 0.1, 0.0], dtype=np.float32),
            "is_format_valid": np.ones(4, dtype=bool),
            "is_action_admissible": np.ones(4, dtype=bool),
            "webshop_task": np.asarray(["Find a red ceramic mug"] * 4, dtype=object),
            "webshop_pre_public_state": np.asarray([state] * 4, dtype=object),
            "webshop_won": np.asarray([True, True, False, False], dtype=bool),
        }

    def __len__(self) -> int:
        return 4


def test_webshop_blpo_end_to_end_smoke() -> None:
    config = BLPOConfig(role_current_batch_center=True, assert_exact_alignment=False)
    estimator = BLPOEstimator(config)
    token_rewards = torch.tensor([[1.0], [0.8], [0.1], [0.0]])
    mask = torch.ones_like(token_rewards)
    advantages, returns, metrics, examples = estimator.compute(
        batch=Batch(),
        token_level_rewards=token_rewards,
        response_mask=mask,
        update=1,
    )
    assert advantages.shape == token_rewards.shape
    assert torch.isfinite(advantages).all()
    assert torch.equal(advantages, returns)
    assert "blpo/coverage/both_exact_role_active" in metrics
    assert examples
