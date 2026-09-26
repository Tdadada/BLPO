from __future__ import annotations

from blpo import BLPOConfig, BLPOMemory
from blpo.algorithm import MomentRecord
from blpo.integration import BLPOEstimator


def test_paper_defaults_enable_both_credit_levels() -> None:
    config = BLPOConfig()
    assert config.layers == ("exact", "role")
    assert config.role_confidence
    assert config.exact_action_balance
    assert config.ema_half_life == 8.0


def test_memory_decay_and_round_trip(tmp_path) -> None:
    memory = BLPOMemory(half_life=8.0)
    memory.commit(
        "role",
        ("progress", "search"),
        MomentRecord(W=4.0, S=2.0, Q=3.0, R=4.0, last_update=1),
    )
    preview = memory.preview("role", ("progress", "search"), update=9)
    assert abs(preview.W - 2.0) < 1e-9

    path = tmp_path / "memory.json"
    memory.save(path)
    restored = BLPOMemory(half_life=8.0)
    restored.load(path)
    assert restored.state_dict() == memory.state_dict()


def test_estimator_checkpoints_memory(tmp_path) -> None:
    estimator = BLPOEstimator(BLPOConfig())
    estimator.memory.commit(
        "role",
        ("progress", "click"),
        MomentRecord(W=1.0, S=0.5, Q=0.25, R=1.0, last_update=3),
    )
    saved = estimator.save(tmp_path)
    assert saved.name == "blpo_memory.json"

    reloaded = BLPOEstimator(BLPOConfig())
    reloaded.load(tmp_path)
    assert reloaded.memory.state_dict() == estimator.memory.state_dict()
