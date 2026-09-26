"""Beyond-Local Policy Optimization credit estimation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from . import returns

from .actions import abstract_action, canonical_exact_action
from .keys import KeyBatch, build_key_batch
from .environments.webshop import (
    ABSTRACTION_VERSIONS,
    progress_action_is_shareable,
    progress_role_is_shareable,
)


LAYER_ORDER = ("exact", "role", "stage")
FAMILY_METRIC_NAMES = {
    "pick_and_place": "place",
    "pick_two_obj": "pick_two",
    "clean": "clean",
    "heat": "hot",
    "cool": "cool",
    "look": "look",
}


def _object_array(values: Iterable[Any]) -> np.ndarray:
    values = list(values)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=False)


def _child_id(value: Any) -> str:
    return hashlib.sha1(_encode(value).encode("utf-8")).hexdigest()


@dataclass
class BLPOConfig:
    """Configuration of the BLPO estimator reported in the paper."""

    layers: tuple[str, ...] = ("exact", "role")
    exact_prior: float = 1.0
    role_prior: float = 0.5
    stage_prior: float = 0.25
    ema_half_life: float = 8.0
    epsilon: float = 1e-6
    mode: str = "mean_std_norm"
    role_estimator: str = "residual_action"
    role_application: str = "parallel"
    role_residual_beta: float = 0.5
    role_scope: str = "all"
    webshop_abstraction: str = "decision_frontier_v4"
    role_current_batch_center: bool = False
    role_read_before_write: bool = False
    role_std_floor: float = 0.0
    role_confidence: bool = True
    role_confidence_tau: float = 0.05
    role_confidence_support_scale: float = 4.0
    role_residual_center: bool = True
    role_use_ema: bool = True
    role_action_abstraction: bool = True
    role_child_balance: bool = True
    exact_action_balance: bool = True
    alfworld_role_version: str = "role_v1"
    all_fail_first_visit: bool = False
    all_fail_first_visit_weight: float = 0.1
    progress_first_visit_backoff: bool = False
    step_advantage_w: float = 1.0
    step_records_dir: str | None = None
    audit_run_seed: int = 0
    step_records_raw_sample_rate: float = 0.002
    assert_exact_alignment: bool = True

    def __post_init__(self) -> None:
        self.layers = tuple(str(item).lower() for item in self.layers)
        unknown = set(self.layers) - set(LAYER_ORDER)
        if unknown:
            raise ValueError(f"unknown BLPO layers: {sorted(unknown)}")
        if "exact" not in self.layers:
            raise ValueError("pilot requires Exact in every configuration")
        if "stage" in self.layers and "role" not in self.layers:
            raise ValueError("Stage requires Role")
        if self.ema_half_life <= 0:
            raise ValueError("ema_half_life must be positive")
        if self.mode not in {"mean_norm", "mean_std_norm"}:
            raise ValueError(f"unsupported BLPO normalization mode: {self.mode}")
        if self.role_estimator not in {"raw", "residual_action"}:
            raise ValueError(f"unsupported Role estimator: {self.role_estimator}")
        if self.role_application not in {"parallel", "local_residual"}:
            raise ValueError(
                f"unsupported Role application: {self.role_application}"
            )
        if self.role_application == "local_residual":
            if self.role_estimator != "residual_action":
                raise ValueError(
                    "local residual Role application requires residual_action"
                )
            if self.layers != ("exact", "role"):
                raise ValueError(
                    "local residual Role application requires Exact+Role only"
                )
            if self.role_current_batch_center:
                raise ValueError(
                    "local residual Role application centers only inside Exact"
                )
        if self.role_residual_beta < 0:
            raise ValueError("role_residual_beta must be non-negative")
        if self.role_estimator == "residual_action" and "stage" in self.layers:
            raise ValueError("Residual Action Role v2 does not support Stage")
        if self.role_scope not in {"all", "grounded"}:
            raise ValueError(f"unsupported Role scope: {self.role_scope}")
        if self.role_scope != "all" and self.role_estimator != "residual_action":
            raise ValueError("selective Role scope currently requires residual_action")
        if self.webshop_abstraction not in ABSTRACTION_VERSIONS:
            raise ValueError(
                f"unsupported WebShop abstraction: {self.webshop_abstraction}"
            )
        if self.role_current_batch_center and self.role_estimator != "residual_action":
            raise ValueError(
                "current-batch Role centering requires residual_action"
            )
        if self.role_read_before_write and self.role_estimator != "residual_action":
            raise ValueError(
                "read-before-write Role timing requires residual_action"
            )
        if self.role_std_floor < 0:
            raise ValueError("role_std_floor must be non-negative")
        if self.role_std_floor > 0 and self.role_estimator != "residual_action":
            raise ValueError(
                "role_std_floor currently requires residual_action"
            )
        if self.role_confidence and self.role_estimator != "residual_action":
            raise ValueError("Role confidence requires residual_action")
        if self.role_confidence and self.mode != "mean_std_norm":
            raise ValueError("Role confidence requires mean_std_norm")
        if self.role_confidence and self.role_std_floor > 0:
            raise ValueError(
                "Role confidence replaces role_std_floor; do not enable both"
            )
        if self.role_confidence_tau <= 0:
            raise ValueError("role_confidence_tau must be positive")
        if self.role_confidence_support_scale <= 0:
            raise ValueError("role_confidence_support_scale must be positive")
        if (
            not self.role_residual_center
            or not self.role_use_ema
            or not self.role_action_abstraction
            or not self.role_child_balance
        ) and self.role_estimator != "residual_action":
            raise ValueError(
                "Role residual, memory, and action controls require residual_action"
            )
        if self.alfworld_role_version not in {"role_v1", "goal_progress_v2"}:
            raise ValueError(
                f"unsupported ALFWorld Role version: {self.alfworld_role_version}"
            )
        if self.all_fail_first_visit_weight < 0:
            raise ValueError("all_fail_first_visit_weight must be non-negative")
        if self.progress_first_visit_backoff:
            if self.all_fail_first_visit:
                raise ValueError(
                    "progress first-visit backoff replaces the legacy all-fail bonus"
                )
            if self.role_estimator != "residual_action" or not self.role_confidence:
                raise ValueError(
                    "progress first-visit backoff requires confidence-calibrated residual Role"
                )
            if self.layers != ("exact", "role"):
                raise ValueError(
                    "progress first-visit backoff requires Exact+Role only"
                )
            if self.role_application != "parallel":
                raise ValueError(
                    "progress first-visit backoff requires parallel Role fusion"
                )
            if self.role_current_batch_center:
                raise ValueError(
                    "progress first-visit backoff requires uncentered confidence calibration"
                )
        if self.step_records_dir is not None:
            d = Path(self.step_records_dir)
            if not d.is_absolute():
                raise ValueError(f"step_records_dir must be absolute: {d}")
        if not 0.0 <= self.step_records_raw_sample_rate <= 1.0:
            raise ValueError("step_records_raw_sample_rate must be in [0, 1]")

    @property
    def decay(self) -> float:
        return float(2.0 ** (-1.0 / self.ema_half_life))

    @property
    def priors(self) -> dict[str, float]:
        return {
            "exact": self.exact_prior,
            "role": self.role_prior,
            "stage": self.stage_prior,
        }


@dataclass
class MomentRecord:
    W: float = 0.0
    S: float = 0.0
    Q: float = 0.0
    R: float = 0.0
    last_update: int = 0
    children: dict[str, float] = field(default_factory=dict)

    def decayed(self, update: int, decay: float) -> "MomentRecord":
        delta = max(0, int(update) - int(self.last_update))
        factor = decay**delta
        children = {
            key: value * factor
            for key, value in self.children.items()
            if value * factor > 1e-10
        }
        return MomentRecord(
            W=self.W * factor,
            S=self.S * factor,
            Q=self.Q * factor,
            R=self.R * factor * factor,
            last_update=int(update),
            children=children,
        )


class BLPOMemory:
    """EMA moments for Role and Stage; Exact never enters this table."""

    VERSION = 1

    def __init__(self, half_life: float = 8.0):
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self.half_life = float(half_life)
        self.decay = float(2.0 ** (-1.0 / self.half_life))
        self.tables: dict[str, dict[str, MomentRecord]] = {"role": {}, "stage": {}}

    def preview(self, layer: str, key: Any, update: int) -> MomentRecord:
        encoded = _encode(key)
        record = self.tables[layer].get(encoded, MomentRecord(last_update=int(update)))
        return record.decayed(update, self.decay)

    def commit(self, layer: str, key: Any, record: MomentRecord) -> None:
        self.tables[layer][_encode(key)] = record

    def key_count(self, layer: str) -> int:
        return len(self.tables[layer])

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "half_life": self.half_life,
            "tables": {
                layer: {key: asdict(record) for key, record in table.items()}
                for layer, table in self.tables.items()
            },
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload.get("version", -1)) != self.VERSION:
            raise ValueError(f"unsupported BLPO memory version: {payload.get('version')}")
        half_life = float(payload["half_life"])
        if abs(half_life - self.half_life) > 1e-12:
            raise ValueError(f"EMA half-life mismatch: checkpoint={half_life} config={self.half_life}")
        self.tables = {"role": {}, "stage": {}}
        for layer in ("role", "stage"):
            for key, raw in payload.get("tables", {}).get(layer, {}).items():
                self.tables[layer][key] = MomentRecord(
                    W=float(raw["W"]),
                    S=float(raw["S"]),
                    Q=float(raw["Q"]),
                    R=float(raw.get("R", 0.0)),
                    last_update=int(raw["last_update"]),
                    children={str(k): float(v) for k, v in raw.get("children", {}).items()},
                )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state_dict(), sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _group_members(keys: Iterable[Any]) -> dict[Any, list[int]]:
    result: dict[Any, list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        result[key].append(idx)
    return result


def _summary(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return 0.0, 0.0, 0.0
    return float(array.mean()), float(np.median(array)), float(np.percentile(array, 90))


def _array_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {f"{prefix}/mean": 0.0, f"{prefix}/std": 0.0, f"{prefix}/max_abs": 0.0}
    return {
        f"{prefix}/mean": float(values.mean()),
        f"{prefix}/std": float(values.std()),
        f"{prefix}/max_abs": float(np.max(np.abs(values))),
    }


def _partition(keys: Iterable[Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(rows) for rows in _group_members(keys).values()))


def _role_shareable_mask(
    key_batch: KeyBatch,
    role_scope: str,
) -> np.ndarray:
    """Return where cross-Exact Role sharing is semantically grounded."""

    mask = np.ones(len(key_batch.role), dtype=bool)
    if role_scope == "all":
        return mask
    for row, (family, role) in enumerate(zip(key_batch.families, key_batch.role)):
        if str(family) != "webshop":
            continue
        page = (
            str(role[1])
            if isinstance(role, tuple) and len(role) > 1
            else "unknown"
        )
        mask[row] = page in {"item_page", "item_sub_page"}
    return mask


def _weighted_current(
    rows: list[int],
    balance_children: tuple[Any, ...],
    history_children: tuple[Any, ...],
    trajectories: np.ndarray,
) -> tuple[dict[int, float], dict[str, float]]:
    """Each child totals one; trajectories and repeated rows split that one."""

    by_child: dict[Any, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_child[balance_children[row]][str(trajectories[row])].append(row)

    row_weights: dict[int, float] = {}
    history_weights: dict[str, float] = defaultdict(float)
    for child_trajectories in by_child.values():
        trajectory_weight = 1.0 / len(child_trajectories)
        for child_rows in child_trajectories.values():
            row_weight = trajectory_weight / len(child_rows)
            for row in child_rows:
                row_weights[row] = row_weight
                history_weights[_child_id(history_children[row])] += row_weight
    return row_weights, dict(history_weights)


def _effective_count(children: dict[str, float]) -> float:
    values = np.asarray(list(children.values()), dtype=np.float64)
    if not values.size or float(np.square(values).sum()) <= 0:
        return 0.0
    return float(np.square(values.sum()) / np.square(values).sum())


def _ema_layer_advantage(
    *,
    layer: str,
    keys: tuple[Any, ...],
    balance_children: tuple[Any, ...],
    history_children: tuple[Any, ...],
    trajectories: np.ndarray,
    rewards: np.ndarray,
    table: BLPOMemory,
    update: int,
    epsilon: float,
    remove_std: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    advantages = np.zeros(len(keys), dtype=np.float64)
    active = np.zeros(len(keys), dtype=bool)
    groups = _group_members(keys)
    current_sizes = []
    current_child_counts = []
    current_max_ratios = []
    current_effective = []
    ema_max_ratios = []
    ema_effective = []

    for key, rows in groups.items():
        row_weights, history_weights = _weighted_current(
            rows, balance_children, history_children, trajectories
        )
        record = table.preview(layer, key, update)
        for row, weight in row_weights.items():
            reward = float(rewards[row])
            record.W += weight
            record.S += weight * reward
            record.Q += weight * reward * reward
            record.R += weight * weight
        for child, weight in history_weights.items():
            record.children[child] = record.children.get(child, 0.0) + weight

        mean = record.S / record.W if record.W > epsilon else 0.0
        centered_ss = max(record.Q - record.S * record.S / record.W, 0.0) if record.W > epsilon else 0.0
        variance_denom = record.W - record.R / record.W if record.W > epsilon else 0.0
        variance = centered_ss / variance_denom if variance_denom > epsilon else 0.0
        std = math.sqrt(variance)
        effective = _effective_count(record.children)
        is_active = effective >= 2.0 - 1e-9 and variance > epsilon * epsilon
        if is_active:
            for row in rows:
                centered = float(rewards[row]) - mean
                advantages[row] = centered if remove_std else centered / (std + epsilon)
                active[row] = True
        table.commit(layer, key, record)

        current_sizes.append(len(rows))
        current_child_count = len(set(balance_children[row] for row in rows))
        current_child_counts.append(current_child_count)
        current_max_ratios.append(1.0 / current_child_count if current_child_count else 0.0)
        current_effective.append(float(current_child_count))
        child_values = np.asarray(list(record.children.values()), dtype=np.float64)
        ema_max_ratios.append(float(child_values.max() / child_values.sum()) if child_values.size else 0.0)
        ema_effective.append(effective)

    size_mean, size_median, size_p90 = _summary(current_sizes)
    child_mean, child_median, child_p90 = _summary(current_child_counts)
    return advantages, active, {
        f"blpo/{layer}/group_count": float(len(groups)),
        f"blpo/{layer}/group_size_mean": size_mean,
        f"blpo/{layer}/group_size_median": size_median,
        f"blpo/{layer}/group_size_p90": size_p90,
        f"blpo/{layer}/distinct_child_mean": child_mean,
        f"blpo/{layer}/distinct_child_median": child_median,
        f"blpo/{layer}/distinct_child_p90": child_p90,
        f"blpo/{layer}/max_child_weight_ratio_mean": float(np.mean(current_max_ratios)) if current_max_ratios else 0.0,
        f"blpo/{layer}/effective_child_count_mean": float(np.mean(current_effective)) if current_effective else 0.0,
        f"blpo/{layer}/ema_max_child_weight_ratio_mean": float(np.mean(ema_max_ratios)) if ema_max_ratios else 0.0,
        f"blpo/{layer}/ema_effective_child_count_mean": float(np.mean(ema_effective)) if ema_effective else 0.0,
        f"blpo/{layer}/ema_key_count": float(table.key_count(layer)),
    }


_RESIDUAL_ACTION_TAG = "residual_action"


def _residual_action_ema_parts(encoded_key: str) -> tuple[str, str] | None:
    try:
        raw = json.loads(encoded_key)
    except (TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or raw[0] != _RESIDUAL_ACTION_TAG
    ):
        return None
    return str(raw[1]), str(raw[2])


def _record_mean(record: MomentRecord, epsilon: float) -> float | None:
    return record.S / record.W if record.W > epsilon else None


def _record_effect_variance(
    record: MomentRecord,
    epsilon: float,
) -> float:
    if record.W <= epsilon:
        return 0.0
    centered_ss = max(record.Q - record.S * record.S / record.W, 0.0)
    variance_denom = record.W - record.R / record.W
    return (
        float(centered_ss / variance_denom)
        if variance_denom > epsilon
        else 0.0
    )


def _harmonic_mean(values: Iterable[float], epsilon: float) -> float:
    positive = [float(value) for value in values if float(value) > epsilon]
    if not positive:
        return 0.0
    return float(len(positive) / sum(1.0 / value for value in positive))


def _role_confidence_components(
    *,
    action_records: dict[str, MomentRecord],
    action: str,
    role_std: float,
    tau: float,
    support_scale: float,
    epsilon: float,
) -> tuple[float, float, float, float, float, float]:
    """Return calibrated advantage and diagnostics for one Role action."""

    action_names = tuple(sorted(action_records))
    if action not in action_records or len(action_names) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    means = {
        name: float(action_records[name].S / action_records[name].W)
        for name in action_names
    }
    supports = {
        name: _effective_count(action_records[name].children)
        for name in action_names
    }
    mean_ses_sq = {
        name: _record_effect_variance(action_records[name], epsilon)
        / max(supports[name], 1.0)
        for name in action_names
    }

    action_count = float(len(action_names))
    centered = means[action] - float(np.mean(list(means.values())))
    other_names = [name for name in action_names if name != action]
    other_support = _harmonic_mean(
        [supports[name] for name in other_names], epsilon
    )
    comparison_support = _harmonic_mean(
        [supports[action], other_support], epsilon
    )
    support_confidence = comparison_support / (
        comparison_support + support_scale
    )
    se_difference_sq = (
        ((action_count - 1.0) / action_count) ** 2
        * mean_ses_sq[action]
        + sum(mean_ses_sq[name] for name in other_names)
        / (action_count * action_count)
    )
    se_difference = math.sqrt(max(se_difference_sq, 0.0))
    denominator = math.sqrt(
        role_std * role_std + se_difference_sq + tau * tau
    )
    calibrated = support_confidence * centered / max(denominator, epsilon)
    raw_z = centered / max(role_std, epsilon)
    shrinkage = min(
        1.0,
        abs(calibrated) / max(abs(raw_z), epsilon),
    )
    return (
        calibrated,
        support_confidence,
        se_difference,
        comparison_support,
        raw_z,
        shrinkage,
    )


def _residual_role_advantage(
    *,
    key_batch: KeyBatch,
    batch: Any,
    rewards: np.ndarray,
    table: BLPOMemory,
    update: int,
    epsilon: float,
    remove_std: bool,
    webshop_abstraction: str = "goal_compact_v1",
    role_current_batch_center: bool = False,
    role_shareable: np.ndarray | None = None,
    role_read_before_write: bool = False,
    role_std_floor: float = 0.0,
    role_confidence: bool = False,
    role_confidence_tau: float = 0.05,
    role_confidence_support_scale: float = 4.0,
    role_residual_center: bool = True,
    role_use_ema: bool = True,
    role_action_abstraction: bool = True,
    role_child_balance: bool = True,
    audit_out: dict[str, np.ndarray] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
]:
    """Aggregate Role-action effects with independently ablatable estimators."""

    row_count = len(key_batch.exact)
    advantages = np.zeros(row_count, dtype=np.float64)
    active = np.zeros(row_count, dtype=bool)
    queried = np.zeros(row_count, dtype=bool)
    role_scores = np.zeros(row_count, dtype=np.float64)
    deltas = np.zeros(row_count, dtype=np.float64)
    eligible_rows = np.zeros(row_count, dtype=bool)
    confidence_support = np.zeros(row_count, dtype=np.float64)
    confidence_se = np.zeros(row_count, dtype=np.float64)
    confidence_comparison_support = np.zeros(row_count, dtype=np.float64)
    confidence_raw_z = np.zeros(row_count, dtype=np.float64)
    confidence_shrinkage = np.zeros(row_count, dtype=np.float64)
    confidence_precenter = np.zeros(row_count, dtype=np.float64)
    action_effective_support = np.zeros(row_count, dtype=np.float64)
    role_group_variance = np.zeros(row_count, dtype=np.float64)
    role_group_size = np.zeros(row_count, dtype=np.int64)
    role_distinct_contexts = np.zeros(row_count, dtype=np.int64)

    if role_shareable is None:
        role_shareable = np.ones(row_count, dtype=bool)
    else:
        role_shareable = np.asarray(role_shareable, dtype=bool)
        if role_shareable.shape != (row_count,):
            raise ValueError("role_shareable mask does not match physical steps")

    text_actions = batch.non_tensor_batch.get("text_action", [""] * row_count)
    gamefiles = batch.non_tensor_batch.get("gamefile", [""] * row_count)
    webshop_tasks = batch.non_tensor_batch.get("webshop_task", [""] * row_count)
    public_states = batch.non_tensor_batch.get(
        "webshop_pre_public_state", [None] * row_count
    )
    observations = batch.non_tensor_batch.get("anchor_obs", [""] * row_count)
    format_valid = batch.non_tensor_batch.get(
        "is_format_valid",
        batch.non_tensor_batch.get(
            "is_action_valid", np.ones(row_count, dtype=bool)
        ),
    )
    admissible = batch.non_tensor_batch.get(
        "is_action_admissible",
        batch.non_tensor_batch.get(
            "is_action_valid", np.ones(row_count, dtype=bool)
        ),
    )
    if role_action_abstraction:
        actions = tuple(
            abstract_action(
                str(text_actions[row]),
                str(gamefiles[row]),
                webshop_abstraction=webshop_abstraction,
                webshop_task=str(webshop_tasks[row]),
                webshop_public_state=public_states[row],
                anchor_obs=str(observations[row]),
                format_valid=bool(format_valid[row]),
                admissible=bool(admissible[row]),
            )
            for row in range(row_count)
        )
    else:
        # Keep Role state unchanged but stop sharing actions across concrete
        # entity arguments. Validity remains part of the key, matching Exact.
        actions = tuple(
            _encode(
                canonical_exact_action(
                    str(text_actions[row]),
                    format_valid=bool(format_valid[row]),
                    admissible=bool(admissible[row]),
                )
            )
            for row in range(row_count)
        )
    if webshop_abstraction in {
        "constraint_progress_v2",
        "constraint_progress_v3",
        "decision_frontier_v4",
    }:
        semantic_gate = np.asarray(
            [
                str(key_batch.families[row]) != "webshop"
                or progress_action_is_shareable(actions[row])
                for row in range(row_count)
            ],
            dtype=bool,
        )
        if webshop_abstraction in {
            "constraint_progress_v3",
            "decision_frontier_v4",
        }:
            semantic_gate &= np.asarray(
                [
                    str(key_batch.families[row]) != "webshop"
                    or progress_role_is_shareable(key_batch.role[row])
                    for row in range(row_count)
                ],
                dtype=bool,
            )
        role_shareable = role_shareable & semantic_gate

    exact_groups = _group_members(key_batch.exact)
    eligible_exact_groups: set[Any] = set()
    current_effects: dict[tuple[str, str, Any, Any], list[float]] = defaultdict(list)

    for exact_child, rows in exact_groups.items():
        shareable_rows = [row for row in rows if role_shareable[row]]
        values = rewards[rows]
        baseline = float(values.mean())
        deltas[rows] = (
            values - baseline if role_residual_center else values
        )
        is_eligible = (
            len(shareable_rows) >= 2
            and len({actions[row] for row in shareable_rows}) >= 2
        )
        if not is_eligible:
            continue
        eligible_exact_groups.add(exact_child)
        eligible_rows[shareable_rows] = True
        for row in shareable_rows:
            role_token = _encode(key_batch.role[row])
            current_effects[
                (
                    role_token,
                    actions[row],
                    exact_child,
                    key_batch.exact_child[row],
                )
            ].append(float(deltas[row]))

    updated_keys = {
        (role_token, action)
        for role_token, action, _, _ in current_effects
    }

    def commit_current_effects() -> None:
        # Aggregate one mean residual per independent concrete context.
        for (
            role_token,
            action,
            _,
            history_child,
        ), values in current_effects.items():
            key = (_RESIDUAL_ACTION_TAG, role_token, action)
            record = table.preview("role", key, update)
            if role_child_balance:
                effect = float(np.mean(values))
                weight = 1.0
                weighted_sum = effect
                weighted_square_sum = effect * effect
            else:
                observations = np.asarray(values, dtype=np.float64)
                weight = float(observations.size)
                weighted_sum = float(observations.sum())
                weighted_square_sum = float(np.square(observations).sum())
            record.W += weight
            record.S += weighted_sum
            record.Q += weighted_square_sum
            record.R += weight
            child = _child_id(history_child)
            record.children[child] = record.children.get(child, 0.0) + weight
            table.commit("role", key, record)

    if not role_use_ema:
        # When memory is disabled, use only current-batch statistics.
        table.tables["role"].clear()

    if not role_read_before_write:
        commit_current_effects()

    current_role_tokens = {
        _encode(key_batch.role[row])
        for row in np.flatnonzero(role_shareable)
    }
    by_role: dict[str, dict[str, MomentRecord]] = defaultdict(dict)
    for encoded_key, stored_record in list(table.tables["role"].items()):
        parts = _residual_action_ema_parts(encoded_key)
        if parts is None:
            continue
        role_token, action = parts
        if role_token not in current_role_tokens:
            continue
        record = stored_record.decayed(update, table.decay)
        table.tables["role"][encoded_key] = record
        if _record_mean(record, epsilon) is not None:
            by_role[role_token][action] = record

    role_groups: dict[Any, list[int]] = defaultdict(list)
    for row in np.flatnonzero(role_shareable):
        role_groups[key_batch.role[int(row)]].append(int(row))

    current_group_sizes: list[int] = []
    current_child_counts: list[int] = []
    current_max_ratios: list[float] = []
    current_effective: list[float] = []
    ema_max_ratios: list[float] = []
    ema_effective: list[float] = []
    supporting_counts: list[float] = []
    pre_center_values: list[float] = []
    post_center_values: list[float] = []
    centered_role_count = 0
    center_deactivated_steps = 0
    role_stds: list[float] = []
    std_floor_applied_count = 0

    for role, rows in role_groups.items():
        role_token = _encode(role)
        action_records = by_role.get(role_token, {})
        action_means = {
            action: float(record.S / record.W)
            for action, record in action_records.items()
            if record.W > epsilon
        }
        means = np.asarray(list(action_means.values()), dtype=np.float64)

        role_child_weights: dict[str, float] = {}
        for record in action_records.values():
            for child, weight in record.children.items():
                # Each Exact child is one support unit for the Role even when
                # it supplied more than one abstract action.
                role_child_weights[child] = max(
                    role_child_weights.get(child, 0.0),
                    float(weight),
                )
        effective = _effective_count(role_child_weights)
        variance = float(np.var(means, ddof=1)) if means.size >= 2 else 0.0
        mean = float(means.mean()) if means.size else 0.0
        std = math.sqrt(max(variance, 0.0))
        context_count = len({key_batch.exact_child[row] for row in rows})
        role_group_variance[rows] = variance
        role_group_size[rows] = len(rows)
        role_distinct_contexts[rows] = context_count
        is_active = effective >= 2.0 - 1e-9 and variance > epsilon * epsilon
        if is_active:
            role_stds.append(std)
            if not remove_std and std < role_std_floor:
                std_floor_applied_count += 1

        for row in rows:
            record = action_records.get(actions[row])
            if record is None or record.W <= epsilon:
                continue
            queried[row] = True
            score = float(record.S / record.W)
            role_scores[row] = score
            action_support = _effective_count(record.children)
            action_effective_support[row] = action_support
            supporting_counts.append(action_support)
            if is_active:
                centered = score - mean
                if remove_std:
                    advantages[row] = centered
                elif role_confidence:
                    (
                        advantages[row],
                        confidence_support[row],
                        confidence_se[row],
                        confidence_comparison_support[row],
                        confidence_raw_z[row],
                        confidence_shrinkage[row],
                    ) = _role_confidence_components(
                        action_records=action_records,
                        action=actions[row],
                        role_std=std,
                        tau=role_confidence_tau,
                        support_scale=role_confidence_support_scale,
                        epsilon=epsilon,
                    )
                else:
                    # Never amplify a small historical action-score spread.
                    # A floor of zero exactly preserves the legacy z-score.
                    scale = max(std, role_std_floor, epsilon)
                    advantages[row] = centered / scale
                confidence_precenter[row] = advantages[row]
                active[row] = True

        center_rows = [row for row in rows if active[row]]
        if role_current_batch_center and center_rows:
            centered_role_count += 1
            before = advantages[center_rows].copy()
            pre_center_values.extend(float(value) for value in before)
            advantages[center_rows] = before - float(before.mean())
            post_center_values.extend(
                float(value) for value in advantages[center_rows]
            )
            if float(np.max(np.abs(advantages[center_rows]))) <= epsilon:
                advantages[center_rows] = 0.0
                active[center_rows] = False
                center_deactivated_steps += len(center_rows)

        current_group_sizes.append(len(rows))
        child_count = len({key_batch.exact[row] for row in rows})
        current_child_counts.append(child_count)
        current_max_ratios.append(1.0 / child_count if child_count else 0.0)
        current_effective.append(float(child_count))
        child_values = np.asarray(list(role_child_weights.values()), dtype=np.float64)
        ema_max_ratios.append(
            float(child_values.max() / child_values.sum())
            if child_values.size and float(child_values.sum()) > 0
            else 0.0
        )
        ema_effective.append(effective)

    size_mean, size_median, size_p90 = _summary(current_group_sizes)
    child_mean, child_median, child_p90 = _summary(current_child_counts)
    support_mean, support_median, support_p90 = _summary(supporting_counts)
    eligible_group_ratio = (
        len(eligible_exact_groups) / len(exact_groups) if exact_groups else 0.0
    )
    shareable_count = int(role_shareable.sum())
    nonshareable = ~role_shareable
    nonshareable_count = int(nonshareable.sum())
    metrics = {
        "blpo/residual_role/use_exact_residual": float(
            role_residual_center
        ),
        "blpo/residual_role/use_ema": float(role_use_ema),
        "blpo/residual_role/use_action_abstraction": float(
            role_action_abstraction
        ),
        "blpo/residual_role/use_child_balance": float(
            role_child_balance
        ),
        "blpo/role/group_count": float(len(role_groups)),
        "blpo/role/group_size_mean": size_mean,
        "blpo/role/group_size_median": size_median,
        "blpo/role/group_size_p90": size_p90,
        "blpo/role/distinct_child_mean": child_mean,
        "blpo/role/distinct_child_median": child_median,
        "blpo/role/distinct_child_p90": child_p90,
        "blpo/role/max_child_weight_ratio_mean": (
            float(np.mean(current_max_ratios)) if current_max_ratios else 0.0
        ),
        "blpo/role/effective_child_count_mean": (
            float(np.mean(current_effective)) if current_effective else 0.0
        ),
        "blpo/role/ema_max_child_weight_ratio_mean": (
            float(np.mean(ema_max_ratios)) if ema_max_ratios else 0.0
        ),
        "blpo/role/ema_effective_child_count_mean": (
            float(np.mean(ema_effective)) if ema_effective else 0.0
        ),
        "blpo/role/ema_key_count": float(table.key_count("role")),
        "blpo/residual_role/eligible_exact_child_ratio": float(
            eligible_group_ratio
        ),
        "blpo/residual_role/eligible_step_ratio": float(
            eligible_rows.mean() if row_count else 0.0
        ),
        "blpo/residual_role/query_coverage": float(
            queried.mean() if row_count else 0.0
        ),
        "blpo/residual_role/shareable_step_ratio": float(
            role_shareable.mean() if row_count else 0.0
        ),
        "blpo/residual_role/query_coverage_shareable": float(
            queried[role_shareable].mean() if shareable_count else 0.0
        ),
        "blpo/residual_role/active_shareable_step_ratio": float(
            active[role_shareable].mean() if shareable_count else 0.0
        ),
        "blpo/residual_role/nonshareable_active_step_ratio": float(
            active[nonshareable].mean() if nonshareable_count else 0.0
        ),
        "blpo/residual_role/nonshareable_query_step_ratio": float(
            queried[nonshareable].mean() if nonshareable_count else 0.0
        ),
        "blpo/residual_role/supporting_exact_child_mean": support_mean,
        "blpo/residual_role/supporting_exact_child_median": support_median,
        "blpo/residual_role/supporting_exact_child_p90": support_p90,
        "blpo/residual_role/delta_mean": float(
            deltas[eligible_rows].mean() if eligible_rows.any() else 0.0
        ),
        "blpo/residual_role/delta_std": float(
            deltas[eligible_rows].std() if eligible_rows.any() else 0.0
        ),
        "blpo/residual_role/score_std": float(
            role_scores[queried].std() if queried.any() else 0.0
        ),
        "blpo/residual_role/other_ratio": float(
            np.mean(np.asarray(actions, dtype=object) == "OTHER")
            if row_count
            else 0.0
        ),
        "blpo/residual_role/current_effect_count": float(
            len(current_effects)
        ),
        "blpo/residual_role/current_action_key_count": float(
            len(updated_keys)
        ),
        "blpo/residual_role/read_before_write": float(
            role_read_before_write
        ),
        "blpo/residual_role/role_std_floor": float(role_std_floor),
        "blpo/residual_role/role_score_std_mean": float(
            np.mean(role_stds) if role_stds else 0.0
        ),
        "blpo/residual_role/role_score_std_median": float(
            np.median(role_stds) if role_stds else 0.0
        ),
        "blpo/residual_role/std_floor_applied_role_count": float(
            std_floor_applied_count
        ),
        "blpo/residual_role/std_floor_applied_role_ratio": float(
            std_floor_applied_count / len(role_stds)
            if role_stds
            else 0.0
        ),
        "blpo/residual_role/batch_center/enabled": float(
            role_current_batch_center
        ),
        "blpo/residual_role/batch_center/role_count": float(
            centered_role_count
        ),
        "blpo/residual_role/batch_center/pre_mean": float(
            np.mean(pre_center_values) if pre_center_values else 0.0
        ),
        "blpo/residual_role/batch_center/post_mean": float(
            np.mean(post_center_values) if post_center_values else 0.0
        ),
        "blpo/residual_role/batch_center/deactivated_step_ratio": float(
            center_deactivated_steps / row_count if row_count else 0.0
        ),
        "blpo/residual_role/unshareable_action_ratio": float(
            1.0 - role_shareable.mean() if row_count else 0.0
        ),
        "blpo/residual_role/confidence/enabled": float(
            role_confidence
        ),
        "blpo/residual_role/confidence/tau": float(
            role_confidence_tau
        ),
        "blpo/residual_role/confidence/support_scale": float(
            role_confidence_support_scale
        ),
        "blpo/residual_role/confidence/support_mean": float(
            confidence_support[active].mean() if active.any() else 0.0
        ),
        "blpo/residual_role/confidence/support_median": float(
            np.median(confidence_support[active]) if active.any() else 0.0
        ),
        "blpo/residual_role/confidence/se_difference_mean": float(
            confidence_se[active].mean() if active.any() else 0.0
        ),
        "blpo/residual_role/confidence/se_difference_median": float(
            np.median(confidence_se[active]) if active.any() else 0.0
        ),
        "blpo/residual_role/confidence/comparison_support_mean": float(
            confidence_comparison_support[active].mean()
            if active.any()
            else 0.0
        ),
        "blpo/residual_role/confidence/raw_z_abs_mean": float(
            np.abs(confidence_raw_z[active]).mean() if active.any() else 0.0
        ),
        "blpo/residual_role/confidence/precenter_abs_mean": float(
            np.abs(confidence_precenter[active]).mean()
            if active.any()
            else 0.0
        ),
        "blpo/residual_role/confidence/final_abs_mean": float(
            np.abs(advantages[active]).mean() if active.any() else 0.0
        ),
        "blpo/residual_role/confidence/shrinkage_mean": float(
            confidence_shrinkage[active].mean() if active.any() else 0.0
        ),
        "blpo/residual_role/confidence/shrinkage_median": float(
            np.median(confidence_shrinkage[active]) if active.any() else 0.0
        ),
    }

    families = np.asarray(key_batch.families, dtype=object)
    actions_array = np.asarray(actions, dtype=object)
    for family in sorted(set(str(item) for item in key_batch.families)):
        mask = families == family
        family_groups = [
            rows
            for rows in exact_groups.values()
            if rows and str(key_batch.families[rows[0]]) == family
        ]
        eligible_family_groups = sum(
            1 for rows in family_groups if key_batch.exact[rows[0]] in eligible_exact_groups
        )
        family_support = [
            _effective_count(by_role[_encode(key_batch.role[row])][actions[row]].children)
            for row in np.flatnonzero(mask & queried)
        ]
        display_family = FAMILY_METRIC_NAMES.get(family, family)
        prefix = f"blpo/residual_role/{display_family}"
        metrics.update(
            {
                f"{prefix}/eligible_exact_child_ratio": (
                    eligible_family_groups / len(family_groups)
                    if family_groups
                    else 0.0
                ),
                f"{prefix}/query_coverage": float(queried[mask].mean()),
                f"{prefix}/delta_std": float(
                    deltas[mask & eligible_rows].std()
                    if np.any(mask & eligible_rows)
                    else 0.0
                ),
                f"{prefix}/score_std": float(
                    role_scores[mask & queried].std()
                    if np.any(mask & queried)
                    else 0.0
                ),
                f"{prefix}/other_ratio": float(
                    np.mean(actions_array[mask] == "OTHER")
                ),
                f"{prefix}/supporting_exact_child_mean": (
                    float(np.mean(family_support)) if family_support else 0.0
                ),
                f"{prefix}/active_step_ratio": float(active[mask].mean()),
            }
        )

    webshop_rows = families == "webshop"
    if np.any(webshop_rows):
        page_labels = np.asarray(
            [
                str(role[1])
                if isinstance(role, tuple) and len(role) > 1
                else "unknown"
                for role in key_batch.role
            ],
            dtype=object,
        )
        for page in sorted(set(page_labels[webshop_rows].tolist())):
            page_mask = webshop_rows & (page_labels == page)
            shared_actions = actions_array[page_mask & role_shareable]
            counts = np.asarray(
                [
                    np.sum(shared_actions == action)
                    for action in sorted(set(shared_actions.tolist()))
                ],
                dtype=np.float64,
            )
            probabilities = counts / counts.sum() if counts.size else counts
            entropy = float(
                -np.sum(probabilities * np.log(probabilities + 1e-12))
            )
            normalized_entropy = (
                entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
            )
            prefix = f"blpo/residual_role/webshop_stage/{page}"
            metrics[f"{prefix}/shareable_ratio"] = float(
                role_shareable[page_mask].mean()
            )
            metrics[f"{prefix}/abstract_action_count"] = float(len(counts))
            metrics[f"{prefix}/abstract_action_top1_ratio"] = float(
                probabilities.max() if probabilities.size else 0.0
            )
            metrics[f"{prefix}/abstract_action_entropy"] = entropy
            metrics[f"{prefix}/abstract_action_entropy_normalized"] = (
                normalized_entropy
            )
            active_page = page_mask & active
            metrics[f"{prefix}/confidence_support_mean"] = float(
                confidence_support[active_page].mean()
                if np.any(active_page)
                else 0.0
            )
            metrics[f"{prefix}/confidence_se_difference_mean"] = float(
                confidence_se[active_page].mean()
                if np.any(active_page)
                else 0.0
            )
            metrics[f"{prefix}/confidence_shrinkage_mean"] = float(
                confidence_shrinkage[active_page].mean()
                if np.any(active_page)
                else 0.0
            )
            metrics[f"{prefix}/confidence_final_abs_mean"] = float(
                np.abs(advantages[active_page]).mean()
                if np.any(active_page)
                else 0.0
            )

    # In lagged mode all current advantages above were obtained from the
    # decayed pre-update table. Only now make this batch visible to update+1.
    if role_read_before_write:
        commit_current_effects()
    metrics["blpo/role/ema_key_count"] = float(table.key_count("role"))
    if audit_out is not None:
        audit_out.update(
            {
                "residual_delta": deltas,
                "eligible": eligible_rows,
                "confidence_support": confidence_support,
                "confidence_se": confidence_se,
                "confidence_comparison_support": confidence_comparison_support,
                "confidence_raw_z": confidence_raw_z,
                "confidence_shrinkage": confidence_shrinkage,
                "role_score": role_scores,
                "role_queried": queried,
                "action_effective_support": action_effective_support,
                "role_group_variance": role_group_variance,
                "role_group_size": role_group_size,
                "role_distinct_contexts": role_distinct_contexts,
            }
        )
    return (
        advantages,
        active,
        metrics,
        role_shareable,
        role_scores,
        queried,
        actions,
        confidence_shrinkage,
    )


def _local_residual_projection(
    *,
    rewards: np.ndarray,
    exact_keys: tuple[Any, ...],
    role_scores: np.ndarray,
    role_reliable: np.ndarray,
    actions: tuple[str, ...],
    beta: float,
    epsilon: float,
    remove_std: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Project Role residuals only inside the current physical Exact group."""

    row_count = len(exact_keys)
    local = np.zeros(row_count, dtype=np.float64)
    correction = np.zeros(row_count, dtype=np.float64)
    role_applied = np.zeros(row_count, dtype=bool)
    role_rescued = np.zeros(row_count, dtype=bool)
    same_action_blocked = np.zeros(row_count, dtype=bool)
    usable_group_count = 0
    rescued_group_count = 0
    distinct_action_counts: list[int] = []
    groups = _group_members(exact_keys)

    for rows in groups.values():
        values = rewards[rows]
        direct = values - float(values.mean())
        reliable_rows = [row for row in rows if role_reliable[row]]
        reliable_actions = {actions[row] for row in reliable_rows}
        score_values = role_scores[reliable_rows]
        usable = (
            len(reliable_rows) >= 2
            and len(reliable_actions) >= 2
            and float(np.var(score_values)) > epsilon * epsilon
        )

        projected = np.zeros(len(rows), dtype=np.float64)
        if usable:
            usable_group_count += 1
            distinct_action_counts.append(len(reliable_actions))
            centered_scores = score_values - float(score_values.mean())
            row_to_offset = {row: offset for offset, row in enumerate(rows)}
            for score_offset, row in enumerate(reliable_rows):
                projected[row_to_offset[row]] = centered_scores[score_offset]
            correction[rows] = beta * projected
            role_applied[reliable_rows] = True
        elif len(reliable_rows) >= 2 and len(reliable_actions) < 2:
            same_action_blocked[rows] = True

        combined = direct + beta * projected
        combined = combined - float(combined.mean())
        if len(rows) <= 1:
            normalized = np.zeros(len(rows), dtype=np.float64)
        elif remove_std:
            normalized = combined
        else:
            std = float(np.std(combined, ddof=1))
            normalized = (
                combined / (std + epsilon)
                if std > epsilon
                else np.zeros(len(rows), dtype=np.float64)
            )
        local[rows] = normalized

        direct_has_signal = float(np.var(values)) > epsilon * epsilon
        local_has_signal = float(np.max(np.abs(normalized))) > epsilon
        if usable and not direct_has_signal and local_has_signal:
            rescued_group_count += 1
            role_rescued[rows] = True

    affected = np.abs(correction) > epsilon
    local_nonzero = np.abs(local) > epsilon
    direct_centered = np.zeros(row_count, dtype=np.float64)
    for rows in groups.values():
        values = rewards[rows]
        direct_centered[rows] = values - float(values.mean())
    sign_rows = (
        (np.abs(direct_centered) > epsilon)
        & (np.abs(local) > epsilon)
        & role_applied
    )
    metrics = {
        "blpo/local_residual/beta": float(beta),
        "blpo/local_residual/exact_group_count": float(len(groups)),
        "blpo/local_residual/usable_group_ratio": float(
            usable_group_count / len(groups) if groups else 0.0
        ),
        "blpo/local_residual/role_applied_step_ratio": float(
            role_applied.mean() if row_count else 0.0
        ),
        "blpo/local_residual/role_nonzero_correction_ratio": float(
            affected.mean() if row_count else 0.0
        ),
        "blpo/local_residual/role_rescued_group_ratio": float(
            rescued_group_count / len(groups) if groups else 0.0
        ),
        "blpo/local_residual/role_rescued_step_ratio": float(
            role_rescued.mean() if row_count else 0.0
        ),
        "blpo/local_residual/same_action_blocked_step_ratio": float(
            same_action_blocked.mean() if row_count else 0.0
        ),
        "blpo/local_residual/no_signal_step_ratio": float(
            (~local_nonzero).mean() if row_count else 0.0
        ),
        "blpo/local_residual/correction_abs_mean": float(
            np.mean(np.abs(correction)) if row_count else 0.0
        ),
        "blpo/local_residual/advantage_abs_mean": float(
            np.mean(np.abs(local)) if row_count else 0.0
        ),
        "blpo/local_residual/correction_to_advantage_ratio": float(
            np.mean(np.abs(correction))
            / (np.mean(np.abs(local)) + epsilon)
            if row_count
            else 0.0
        ),
        "blpo/local_residual/sign_flip_ratio": float(
            np.mean(
                np.sign(direct_centered[sign_rows])
                != np.sign(local[sign_rows])
            )
            if np.any(sign_rows)
            else 0.0
        ),
        "blpo/local_residual/distinct_known_action_mean": float(
            np.mean(distinct_action_counts) if distinct_action_counts else 0.0
        ),
    }
    return local, correction, role_applied, metrics


def _conflict(a: np.ndarray, b: np.ndarray) -> float:
    eligible = (np.abs(a) > 0) & (np.abs(b) > 0)
    if not eligible.any():
        return 0.0
    return float(np.mean(np.sign(a[eligible]) != np.sign(b[eligible])))


def _pearson(a: np.ndarray, b: np.ndarray, epsilon: float) -> float:
    eligible = (np.abs(a) > epsilon) & (np.abs(b) > epsilon)
    if int(eligible.sum()) < 2:
        return 0.0
    a_values = a[eligible]
    b_values = b[eligible]
    if float(np.std(a_values)) <= epsilon or float(np.std(b_values)) <= epsilon:
        return 0.0
    return float(np.corrcoef(a_values, b_values)[0, 1])


def _conditional_first_visit_advantage(
    *,
    group_uids: Iterable[Any],
    traj_uids: Iterable[Any],
    state_keys: Iterable[Any],
    env_rewards: Iterable[Any],
    admissible: Iterable[Any],
    epsilon: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Break all-fail task-group ties using within-trajectory state coverage."""

    group_uids = tuple(str(item) for item in group_uids)
    traj_uids = tuple(str(item) for item in traj_uids)
    state_keys = tuple(state_keys)
    env_rewards = np.asarray(list(env_rewards), dtype=np.float64).reshape(-1)
    admissible = np.asarray(list(admissible), dtype=bool).reshape(-1)
    row_count = len(group_uids)
    if not (
        len(traj_uids) == row_count
        and len(state_keys) == row_count
        and env_rewards.size == row_count
        and admissible.size == row_count
    ):
        raise ValueError("first-visit fallback metadata has inconsistent lengths")

    rows_by_group: dict[str, list[int]] = defaultdict(list)
    rows_by_traj: dict[str, list[int]] = defaultdict(list)
    for row, (group_uid, traj_uid) in enumerate(zip(group_uids, traj_uids)):
        rows_by_group[group_uid].append(row)
        rows_by_traj[traj_uid].append(row)

    row_advantage = np.zeros(row_count, dtype=np.float64)
    trajectory_scores: list[float] = []
    active_trajectory_scores: list[float] = []
    all_fail_group_count = 0
    active_group_count = 0
    zero_variance_group_count = 0
    malformed_group_count = 0

    for rows in rows_by_group.values():
        group_trajectories = list(dict.fromkeys(traj_uids[row] for row in rows))
        if len(group_trajectories) != 8:
            malformed_group_count += 1
            continue
        if any(
            np.any(env_rewards[rows_by_traj[traj_uid]] > 0.0)
            for traj_uid in group_trajectories
        ):
            continue
        all_fail_group_count += 1

        scores = []
        for traj_uid in group_trajectories:
            traj_rows = rows_by_traj[traj_uid]
            valid_rows = [row for row in traj_rows if admissible[row]]
            if not valid_rows:
                score = 0.0
            else:
                score = len({_encode(state_keys[row]) for row in valid_rows}) / len(valid_rows)
            scores.append(float(score))
            trajectory_scores.append(float(score))

        score_array = np.asarray(scores, dtype=np.float64)
        std = float(np.std(score_array, ddof=1)) if len(score_array) > 1 else 0.0
        if not np.isfinite(std) or std <= epsilon:
            zero_variance_group_count += 1
            continue
        normalized = (score_array - float(score_array.mean())) / (std + epsilon)
        active_group_count += 1
        active_trajectory_scores.extend(scores)
        for traj_uid, advantage in zip(group_trajectories, normalized):
            row_advantage[rows_by_traj[traj_uid]] = float(advantage)

    group_count = len(rows_by_group)
    active_rows = np.abs(row_advantage) > epsilon
    metrics = {
        "blpo/first_visit/enabled": 1.0,
        "blpo/first_visit/group_count": float(group_count),
        "blpo/first_visit/all_fail_group_count": float(all_fail_group_count),
        "blpo/first_visit/all_fail_group_ratio": float(all_fail_group_count / max(1, group_count)),
        "blpo/first_visit/active_group_count": float(active_group_count),
        "blpo/first_visit/active_group_ratio": float(active_group_count / max(1, group_count)),
        "blpo/first_visit/zero_variance_group_count": float(zero_variance_group_count),
        "blpo/first_visit/malformed_group_count": float(malformed_group_count),
        "blpo/first_visit/nonzero_step_ratio": float(active_rows.mean()) if row_count else 0.0,
        "blpo/first_visit/score_mean": float(np.mean(trajectory_scores)) if trajectory_scores else 0.0,
        "blpo/first_visit/score_min": float(np.min(trajectory_scores)) if trajectory_scores else 0.0,
        "blpo/first_visit/score_max": float(np.max(trajectory_scores)) if trajectory_scores else 0.0,
        "blpo/first_visit/active_score_mean": float(np.mean(active_trajectory_scores)) if active_trajectory_scores else 0.0,
        "blpo/first_visit/advantage_abs_mean": float(np.mean(np.abs(row_advantage))) if row_count else 0.0,
    }
    return row_advantage, metrics


def _progress_first_visit_advantage(
    *,
    group_uids: Iterable[Any],
    traj_uids: Iterable[Any],
    progress_keys: Iterable[Any],
    admissible: Iterable[Any],
    epsilon: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize target-progress coverage across every eight-rollout game."""

    group_uids = tuple(str(item) for item in group_uids)
    traj_uids = tuple(str(item) for item in traj_uids)
    progress_keys = tuple(progress_keys)
    admissible = np.asarray(list(admissible), dtype=bool).reshape(-1)
    row_count = len(group_uids)
    if not (
        len(traj_uids) == row_count
        and len(progress_keys) == row_count
        and admissible.size == row_count
    ):
        raise ValueError("progress first-visit metadata has inconsistent lengths")

    rows_by_group: dict[str, list[int]] = defaultdict(list)
    rows_by_traj: dict[str, list[int]] = defaultdict(list)
    for row, (group_uid, traj_uid) in enumerate(zip(group_uids, traj_uids)):
        rows_by_group[group_uid].append(row)
        rows_by_traj[traj_uid].append(row)

    row_advantage = np.zeros(row_count, dtype=np.float64)
    trajectory_scores: list[float] = []
    active_scores: list[float] = []
    active_group_count = 0
    zero_variance_group_count = 0
    malformed_group_count = 0

    for rows in rows_by_group.values():
        group_trajectories = list(dict.fromkeys(traj_uids[row] for row in rows))
        if len(group_trajectories) != 8:
            malformed_group_count += 1
            continue

        scores = []
        for traj_uid in group_trajectories:
            valid_rows = [
                row for row in rows_by_traj[traj_uid] if admissible[row]
            ]
            score = (
                len({_encode(progress_keys[row]) for row in valid_rows})
                / len(valid_rows)
                if valid_rows
                else 0.0
            )
            scores.append(float(score))
            trajectory_scores.append(float(score))

        score_array = np.asarray(scores, dtype=np.float64)
        std = float(np.std(score_array, ddof=1))
        if not np.isfinite(std) or std <= epsilon:
            zero_variance_group_count += 1
            continue
        normalized = (score_array - float(score_array.mean())) / (std + epsilon)
        active_group_count += 1
        active_scores.extend(scores)
        for traj_uid, advantage in zip(group_trajectories, normalized):
            valid_rows = [
                row for row in rows_by_traj[traj_uid] if admissible[row]
            ]
            row_advantage[valid_rows] = float(advantage)

    group_count = len(rows_by_group)
    active_rows = np.abs(row_advantage) > epsilon
    metrics = {
        "blpo/first_visit/enabled": 1.0,
        "blpo/first_visit/progress_role_source": 1.0,
        "blpo/first_visit/group_count": float(group_count),
        "blpo/first_visit/all_fail_group_count": 0.0,
        "blpo/first_visit/all_fail_group_ratio": 0.0,
        "blpo/first_visit/active_group_count": float(active_group_count),
        "blpo/first_visit/active_group_ratio": float(
            active_group_count / max(1, group_count)
        ),
        "blpo/first_visit/zero_variance_group_count": float(
            zero_variance_group_count
        ),
        "blpo/first_visit/malformed_group_count": float(
            malformed_group_count
        ),
        "blpo/first_visit/nonzero_step_ratio": float(
            active_rows.mean()
        ) if row_count else 0.0,
        "blpo/first_visit/score_mean": float(
            np.mean(trajectory_scores)
        ) if trajectory_scores else 0.0,
        "blpo/first_visit/score_min": float(
            np.min(trajectory_scores)
        ) if trajectory_scores else 0.0,
        "blpo/first_visit/score_max": float(
            np.max(trajectory_scores)
        ) if trajectory_scores else 0.0,
        "blpo/first_visit/active_score_mean": float(
            np.mean(active_scores)
        ) if active_scores else 0.0,
        "blpo/first_visit/advantage_abs_mean": float(
            np.mean(np.abs(row_advantage))
        ) if row_count else 0.0,
    }
    return row_advantage, metrics


def _exact_action_balanced_advantage(
    *,
    key_batch: KeyBatch,
    batch: Any,
    rewards: np.ndarray,
    epsilon: float,
    remove_std: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Share credit for identical actions on the original Exact-state scale."""

    row_count = len(key_batch.exact)
    advantages = np.zeros(row_count, dtype=np.float64)
    active = np.zeros(row_count, dtype=bool)
    text_actions = batch.non_tensor_batch.get("text_action", [""] * row_count)
    format_valid = batch.non_tensor_batch.get(
        "is_format_valid",
        batch.non_tensor_batch.get(
            "is_action_valid", np.ones(row_count, dtype=bool)
        ),
    )
    admissible = batch.non_tensor_batch.get(
        "is_action_admissible",
        batch.non_tensor_batch.get(
            "is_action_valid", np.ones(row_count, dtype=bool)
        ),
    )
    action_keys = tuple(
        canonical_exact_action(
            str(text_actions[row]),
            format_valid=bool(format_valid[row]),
            admissible=bool(admissible[row]),
        )
        for row in range(row_count)
    )

    unique_counts: list[int] = []
    repeated_return_stds: list[float] = []
    duplicate_rows = 0
    active_groups = 0
    for rows in _group_members(key_batch.exact).values():
        by_action: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in rows:
            by_action[action_keys[row]].append(row)
        unique_counts.append(len(by_action))
        duplicate_rows += len(rows) - len(by_action)
        action_scores = {
            action: float(np.mean(rewards[action_rows]))
            for action, action_rows in by_action.items()
        }
        for action_rows in by_action.values():
            if len(action_rows) > 1:
                repeated_return_stds.append(
                    float(np.std(rewards[action_rows], ddof=0))
                )
        occurrence_values = np.asarray(rewards[rows], dtype=np.float64)
        variance = (
            float(np.var(occurrence_values, ddof=1))
            if occurrence_values.size >= 2
            else 0.0
        )
        if variance <= epsilon * epsilon:
            continue
        mean = float(occurrence_values.mean())
        scale = math.sqrt(variance)
        centered_scores = {
            action: score - mean for action, score in action_scores.items()
        }
        if not any(abs(value) > epsilon for value in centered_scores.values()):
            continue
        active_groups += 1
        for action, action_rows in by_action.items():
            centered = centered_scores[action]
            value = centered if remove_std else centered / (scale + epsilon)
            advantages[action_rows] = value
            active[action_rows] = True

    count_mean, count_median, count_p90 = _summary(unique_counts)
    metrics = {
        "blpo/exact_action/enabled": 1.0,
        "blpo/exact_action/unique_action_mean": count_mean,
        "blpo/exact_action/unique_action_median": count_median,
        "blpo/exact_action/unique_action_p90": count_p90,
        "blpo/exact_action/duplicate_step_ratio": (
            float(duplicate_rows / row_count) if row_count else 0.0
        ),
        "blpo/exact_action/active_group_ratio": (
            float(active_groups / len(unique_counts)) if unique_counts else 0.0
        ),
        "blpo/exact_action/repeated_action_return_std_mean": (
            float(np.mean(repeated_return_stds))
            if repeated_return_stds
            else 0.0
        ),
    }
    return advantages, active, metrics


def _audit_hash(value: Any) -> str:
    return hashlib.sha1(_encode(value).encode("utf-8")).hexdigest()


def _audit_row(values: Any, row: int, default: Any = None) -> Any:
    if values is None:
        return default
    try:
        value = values[row]
    except (IndexError, KeyError, TypeError):
        return default
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_step_records(
    *, output_dir: str, run_seed: int, raw_sample_rate: float, update: int,
    batch: Any, key_batch: KeyBatch, rewards: np.ndarray, exact: np.ndarray,
    exact_active: np.ndarray, progress_raw: np.ndarray,
    progress_final: np.ndarray, progress_active: np.ndarray,
    role_shareable: np.ndarray, role_actions: tuple[str, ...],
    role_audit: dict[str, np.ndarray],
) -> int:
    """Atomically persist one Parquet partition per policy update."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=Path(output_dir)
    if not root.is_absolute():
        raise ValueError(f"step-record output must be absolute: {root}")
    part=root/f"u{int(update):04d}"
    dst=part/f"part-{int(update):04d}.parquet"
    if dst.exists(): raise FileExistsError(f"audit partition exists: {dst}")
    part.mkdir(parents=True,exist_ok=True)
    nt=batch.non_tensor_batch
    n=len(key_batch.exact)
    traj=tuple(str(x) for x in nt["traj_uid"])
    uid=tuple(str(x) for x in nt["uid"])
    games=nt.get("gamefile",[""]*n)
    tasks=nt.get("webshop_task",[""]*n)
    obs=nt.get("anchor_obs",[""]*n)
    acts=nt.get("text_action",[""]*n)
    fmt=nt.get("is_format_valid",nt.get("is_action_valid",np.ones(n,dtype=bool)))
    adm=nt.get("is_action_admissible",nt.get("is_action_valid",np.ones(n,dtype=bool)))
    ep=nt.get("episode_rewards")
    imm=nt.get("rewards")
    won=nt.get("webshop_won")
    task_ids=[]
    for i,fam in enumerate(key_batch.families):
        if str(fam)=="webshop":
            child=key_batch.exact_child[i]
            task_ids.append(str(child[1]) if isinstance(child,tuple) and len(child)>1 else _audit_hash(("webshop_task",str(_audit_row(tasks,i,"")))))
        else:
            task_ids.append(_audit_hash(("alfworld_task",str(_audit_row(games,i,"")))))
    role_hash=[_audit_hash(x) for x in key_batch.role]
    ctx_hash=[_audit_hash((task_ids[i],key_batch.exact_child[i])) for i in range(n)]
    by_role=defaultdict(list)
    for i,k in enumerate(role_hash): by_role[k].append(i)
    group_count=np.zeros(n,dtype=np.int64)
    distinct_tasks=np.zeros(n,dtype=np.int64)
    distinct_contexts=np.zeros(n,dtype=np.int64)
    for rows in by_role.values():
        group_count[rows]=len(rows)
        distinct_tasks[rows]=len({task_ids[i] for i in rows})
        distinct_contexts[rows]=len({ctx_hash[i] for i in rows})
    step_idx=[]
    counters=defaultdict(int)
    for t in traj:
        step_idx.append(counters[t]); counters[t]+=1
    records=[]
    for i in range(n):
        fam=str(key_batch.families[i])
        episode_return=float(_audit_row(ep,i,0.0))
        w=_audit_row(won,i)
        success=bool(w) if w is not None else episode_return >= (10.0 if fam=="webshop" else 1.0)
        exact_action=canonical_exact_action(str(_audit_row(acts,i,"")),format_valid=bool(_audit_row(fmt,i,False)),admissible=bool(_audit_row(adm,i,False)))
        token=_audit_hash((int(run_seed),int(update),traj[i],step_idx[i]))
        keep=int(token[:12],16)/float(16**12)<float(raw_sample_rate)
        records.append({
          "env":"webshop" if fam=="webshop" else "alfworld","run_seed":int(run_seed),
          "update":int(update),"task_id":task_ids[i],"trajectory_id":traj[i],
          "group_uid":uid[i],"step_idx":int(step_idx[i]),"episode_success":success,
          "episode_return":episode_return,"discounted_return":float(rewards[i]),
          "immediate_reward":float(_audit_row(imm,i,0.0)),"task_category":fam,
          "exact_state_key":_audit_hash(key_batch.exact[i]),
          "normalized_action_key":_audit_hash(exact_action),
          "progress_state_key":role_hash[i],"progress_action_key":_audit_hash(role_actions[i]),
          "concrete_context_key":ctx_hash[i],"local_advantage":float(exact[i]),
          "progress_advantage_raw":float(progress_raw[i]),
          "progress_advantage_final":float(progress_final[i]),
          "confidence":float(role_audit["confidence_shrinkage"][i]),
          "confidence_support":float(role_audit["confidence_support"][i]),
          "confidence_se":float(role_audit["confidence_se"][i]),
          "effective_support":float(role_audit["action_effective_support"][i]),
          "comparison_support":float(role_audit["confidence_comparison_support"][i]),
          "group_variance":float(role_audit["role_group_variance"][i]),
          "shareable":bool(role_shareable[i]),"local_active":bool(exact_active[i]),
          "progress_active":bool(progress_active[i]),
          "progress_queried":bool(role_audit["role_queried"][i]),
          "residual_eligible":bool(role_audit["eligible"][i]),
          "residual_delta":float(role_audit["residual_delta"][i]),
          "role_score":float(role_audit["role_score"][i]),
          "group_count":int(group_count[i]),"distinct_tasks":int(distinct_tasks[i]),
          "distinct_contexts":int(distinct_contexts[i]),
          "ema_group_size":int(role_audit["role_group_size"][i]),
          "ema_distinct_contexts":int(role_audit["role_distinct_contexts"][i]),
          "raw_observation":str(_audit_row(obs,i,"")) if keep else None,
          "raw_action":str(_audit_row(acts,i,"")) if keep else None,
          "raw_task":str(_audit_row(tasks if fam=="webshop" else games,i,"")) if keep else None,
        })
    table=pa.Table.from_pylist(records)
    tmp=dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    try:
        pq.write_table(table,tmp,compression="zstd"); tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return len(records)


def compute_blpo_advantage(
    *,
    batch: Any,
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    table: BLPOMemory,
    config: BLPOConfig,
    update: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], tuple[dict[str, Any], ...]]:
    """Compute BLPO trajectory, local, and task-progress advantages."""

    remove_std = config.mode == "mean_norm"
    key_batch: KeyBatch = build_key_batch(
        batch,
        config.webshop_abstraction,
        config.alfworld_role_version,
    )
    role_shareable = _role_shareable_mask(key_batch, config.role_scope)
    exact_keys = _object_array(key_batch.exact)
    uid = batch.non_tensor_batch["uid"]
    traj_uid = batch.non_tensor_batch["traj_uid"]

    episode = returns.episode_norm_reward(
        token_level_rewards,
        response_mask,
        uid,
        traj_uid,
        config.epsilon,
        remove_std,
    )
    base_episode = episode.clone()
    first_visit = np.zeros(len(key_batch.exact), dtype=np.float64)
    first_visit_metrics: dict[str, float] = {
        "blpo/first_visit/enabled": 0.0,
        "blpo/first_visit/nonzero_step_ratio": 0.0,
        "blpo/first_visit/advantage_abs_mean": 0.0,
    }
    progress_prior_component = np.zeros_like(first_visit)
    if config.all_fail_first_visit:
        raw_env_rewards = batch.non_tensor_batch.get("rewards")
        if raw_env_rewards is None:
            raise RuntimeError("first-visit fallback requires raw environment rewards")
        admissible = batch.non_tensor_batch.get(
            "is_action_admissible",
            batch.non_tensor_batch.get("is_action_valid"),
        )
        if admissible is None:
            raise RuntimeError("first-visit fallback requires action validity metadata")
        first_visit, first_visit_metrics = _conditional_first_visit_advantage(
            group_uids=uid,
            traj_uids=traj_uid,
            state_keys=key_batch.exact_child,
            env_rewards=raw_env_rewards,
            admissible=admissible,
            epsilon=config.epsilon,
        )
        first_visit_tensor = torch.as_tensor(
            first_visit,
            dtype=episode.dtype,
            device=episode.device,
        ).unsqueeze(-1) * response_mask
        episode = episode + config.all_fail_first_visit_weight * first_visit_tensor
    elif config.progress_first_visit_backoff:
        admissible = batch.non_tensor_batch.get(
            "is_action_admissible",
            batch.non_tensor_batch.get("is_action_valid"),
        )
        if admissible is None:
            raise RuntimeError(
                "progress first-visit backoff requires action validity metadata"
            )
        admissible = np.asarray(admissible, dtype=bool) & role_shareable
        first_visit, first_visit_metrics = _progress_first_visit_advantage(
            group_uids=uid,
            traj_uids=traj_uid,
            progress_keys=key_batch.role,
            admissible=admissible,
            epsilon=config.epsilon,
        )
    rewards = step_rewards.detach().cpu().numpy().astype(np.float64)
    if config.exact_action_balance:
        exact, exact_active, exact_method_metrics = (
            _exact_action_balanced_advantage(
                key_batch=key_batch,
                batch=batch,
                rewards=rewards,
                epsilon=config.epsilon,
                remove_std=remove_std,
            )
        )
        exact_tokens = (
            torch.as_tensor(
                exact,
                dtype=step_rewards.dtype,
                device=response_mask.device,
            ).unsqueeze(-1)
            * response_mask
        )
    else:
        exact_tokens = returns.step_norm_reward(
            step_rewards,
            response_mask,
            exact_keys,
            config.epsilon,
            remove_std,
        )
        mask_count = response_mask.sum(dim=-1).clamp_min(1)
        exact = (
            exact_tokens.sum(dim=-1) / mask_count
        ).detach().cpu().numpy().astype(np.float64)
        exact_active = np.zeros(len(key_batch.exact), dtype=bool)
        for rows in _group_members(key_batch.exact).values():
            values = rewards[rows]
            active_group = len(rows) > 1 and float(np.var(values)) > 0.0
            if active_group:
                exact_active[rows] = True
        exact_method_metrics = {
            "blpo/exact_action/enabled": 0.0,
        }

    exact_groups = _group_members(key_batch.exact)
    exact_traj_counts = []
    exact_sizes = []
    for rows in exact_groups.values():
        exact_sizes.append(len(rows))
        exact_traj_counts.append(len({str(traj_uid[row]) for row in rows}))

    layer_adv = {
        "exact": exact,
        "role": np.zeros_like(exact),
        "stage": np.zeros_like(exact),
    }
    layer_active = {
        "exact": exact_active,
        "role": np.zeros_like(exact_active),
        "stage": np.zeros_like(exact_active),
    }
    metrics: dict[str, float] = dict(first_visit_metrics)
    metrics.update(exact_method_metrics)
    history_role = np.zeros_like(exact)
    history_role_active = np.zeros_like(exact_active)
    role_reliability = np.zeros_like(exact)
    residual_details: tuple[
        np.ndarray, np.ndarray, tuple[str, ...]
    ] | None = None
    role_actions: tuple[str, ...] = ("UNAVAILABLE",) * len(exact)
    role_audit = {key: np.zeros(len(exact), dtype=np.float64) for key in (
        "residual_delta", "eligible", "confidence_support", "confidence_se",
        "confidence_comparison_support", "confidence_raw_z", "confidence_shrinkage",
        "role_score", "role_queried", "action_effective_support",
        "role_group_variance", "role_group_size", "role_distinct_contexts")}
    size_mean, size_median, size_p90 = _summary(exact_sizes)
    traj_mean, traj_median, traj_p90 = _summary(exact_traj_counts)
    metrics.update(
        {
            "blpo/exact/group_count": float(len(exact_groups)),
            "blpo/exact/group_size_mean": size_mean,
            "blpo/exact/group_size_median": size_median,
            "blpo/exact/group_size_p90": size_p90,
            "blpo/exact/distinct_trajectory_mean": traj_mean,
            "blpo/exact/distinct_trajectory_median": traj_median,
            "blpo/exact/distinct_trajectory_p90": traj_p90,
        }
    )

    if "role" in config.layers:
        if config.role_estimator == "raw":
            role, role_active, role_metrics = _ema_layer_advantage(
                layer="role",
                keys=key_batch.role,
                balance_children=key_batch.exact,
                history_children=key_batch.exact_child,
                trajectories=traj_uid,
                rewards=rewards,
                table=table,
                update=update,
                epsilon=config.epsilon,
                remove_std=remove_std,
            )
        else:
            (
                role,
                role_active,
                role_metrics,
                role_shareable,
                role_scores,
                role_queried,
                role_actions,
                role_reliability,
            ) = _residual_role_advantage(
                key_batch=key_batch,
                batch=batch,
                rewards=rewards,
                table=table,
                update=update,
                epsilon=config.epsilon,
                remove_std=remove_std,
                webshop_abstraction=config.webshop_abstraction,
                role_current_batch_center=config.role_current_batch_center,
                role_shareable=role_shareable,
                role_read_before_write=config.role_read_before_write,
                role_std_floor=config.role_std_floor,
                role_confidence=config.role_confidence,
                role_confidence_tau=config.role_confidence_tau,
                role_confidence_support_scale=config.role_confidence_support_scale,
                role_residual_center=config.role_residual_center,
                role_use_ema=config.role_use_ema,
                role_action_abstraction=config.role_action_abstraction,
                role_child_balance=config.role_child_balance,
                audit_out=role_audit,
            )
            residual_details = (
                role_scores,
                role_queried & role_active,
                role_actions,
            )
        history_role = role.copy()
        history_role_active = role_active.copy()
        if config.progress_first_visit_backoff:
            role_reliability = np.clip(role_reliability, 0.0, 1.0)
            progress_prior_component = (
                1.0 - role_reliability
            ) * first_visit
            role = history_role + progress_prior_component
            role_active = role_active | (
                np.abs(progress_prior_component) > config.epsilon
            )
            metrics.update(
                {
                    "blpo/shared/history_abs_mean": float(
                        np.mean(np.abs(history_role))
                    ),
                    "blpo/shared/reliability_mean": float(
                        np.mean(role_reliability)
                    ),
                    "blpo/shared/reliability_active_mean": float(
                        np.mean(role_reliability[history_role_active])
                        if history_role_active.any()
                        else 0.0
                    ),
                    "blpo/shared/progress_prior_raw_abs_mean": float(
                        np.mean(np.abs(first_visit))
                    ),
                    "blpo/shared/progress_prior_component_abs_mean": float(
                        np.mean(np.abs(progress_prior_component))
                    ),
                    "blpo/shared/progress_prior_nonzero_ratio": float(
                        np.mean(
                            np.abs(progress_prior_component) > config.epsilon
                        )
                    ),
                    "blpo/shared/advantage_abs_mean": float(
                        np.mean(np.abs(role))
                    ),
                    "blpo/shared/positive_ratio": float(
                        np.mean(role > config.epsilon)
                    ),
                    "blpo/shared/negative_ratio": float(
                        np.mean(role < -config.epsilon)
                    ),
                }
            )
        layer_adv["role"] = role
        layer_active["role"] = role_active
        metrics.update(role_metrics)
    if "stage" in config.layers:
        stage, stage_active, stage_metrics = _ema_layer_advantage(
            layer="stage",
            keys=key_batch.stage,
            balance_children=key_batch.role,
            history_children=key_batch.role,
            trajectories=traj_uid,
            rewards=rewards,
            table=table,
            update=update,
            epsilon=config.epsilon,
            remove_std=remove_std,
        )
        layer_adv["stage"] = stage
        layer_active["stage"] = stage_active
        metrics.update(stage_metrics)

    for layer in ("role", "stage"):
        for suffix in (
            "group_count",
            "group_size_mean",
            "group_size_median",
            "group_size_p90",
            "distinct_child_mean",
            "distinct_child_median",
            "distinct_child_p90",
            "max_child_weight_ratio_mean",
            "effective_child_count_mean",
            "ema_max_child_weight_ratio_mean",
            "ema_effective_child_count_mean",
            "ema_key_count",
        ):
            metrics.setdefault(f"blpo/{layer}/{suffix}", 0.0)

    priors = config.priors
    weights = {layer: np.zeros_like(exact) for layer in LAYER_ORDER}
    if config.role_application == "local_residual":
        if residual_details is None:
            raise RuntimeError("local residual projection has no Role scores")
        role_scores, role_reliable, role_actions = residual_details
        micro, role_correction, role_applied, local_metrics = (
            _local_residual_projection(
                rewards=rewards,
                exact_keys=key_batch.exact,
                role_scores=role_scores,
                role_reliable=role_reliable,
                actions=role_actions,
                beta=config.role_residual_beta,
                epsilon=config.epsilon,
                remove_std=remove_std,
            )
        )
        # Role is a local correction diagnostic, never an independent
        # advantage path. The final micro signal is normalized once above.
        layer_adv["role"] = role_correction
        layer_active["role"] = role_applied
        weights["exact"][np.abs(micro) > config.epsilon] = 1.0
        weights["role"][role_applied] = config.role_residual_beta
        metrics.update(local_metrics)
    else:
        micro = np.zeros_like(exact)
        for row in range(len(exact)):
            denominator = sum(
                priors[layer]
                for layer in config.layers
                if layer_active[layer][row]
            )
            if denominator <= 0:
                continue
            for layer in config.layers:
                if layer_active[layer][row]:
                    weights[layer][row] = priors[layer] / denominator
                    micro[row] += (
                        weights[layer][row] * layer_adv[layer][row]
                    )

    # Production response masks are integer tensors. Casting the normalized
    # micro advantage to the mask dtype would silently truncate it to {-1, 0,
    # 1}; keep the floating dtype used by the local step advantage instead.
    micro_tensor = torch.as_tensor(micro, dtype=exact_tokens.dtype, device=response_mask.device)
    micro_tokens = micro_tensor.unsqueeze(-1) * response_mask
    scores = episode + config.step_advantage_w * micro_tokens

    finite = np.isfinite(np.stack([layer_adv[layer] for layer in LAYER_ORDER] + [micro])).all(axis=0)
    if not finite.all() or not torch.isfinite(scores).all():
        raise RuntimeError("BLPO advantage contains NaN or Inf")

    n = max(1, len(exact))
    for layer in LAYER_ORDER:
        active = layer_active[layer]
        metrics[f"blpo/{layer}/active_step_ratio"] = float(active.mean())
        metrics[f"blpo/{layer}/nonzero_advantage_ratio"] = float(np.mean(np.abs(layer_adv[layer]) > 0))
        metrics[f"blpo/weight/{layer}_mean"] = float(weights[layer].mean())
        metrics.update(_array_stats(f"blpo/advantage/{layer}", layer_adv[layer]))
    metrics.update(_array_stats("blpo/advantage/micro", micro))
    base_episode_row = (
        base_episode.sum(dim=-1) / response_mask.sum(dim=-1).clamp_min(1)
    ).detach().cpu().numpy()
    fallback_scaled = (
        progress_prior_component
        if config.progress_first_visit_backoff
        else config.all_fail_first_visit_weight * first_visit
    )
    metrics["blpo/first_visit/weight"] = float(
        0.0
        if config.progress_first_visit_backoff
        else config.all_fail_first_visit_weight
    )
    metrics["blpo/first_visit/confidence_backoff"] = float(
        config.progress_first_visit_backoff
    )
    metrics["blpo/first_visit/scaled_abs_mean"] = float(np.mean(np.abs(fallback_scaled)))
    metrics["blpo/first_visit/scaled_to_episode_abs_ratio"] = float(
        np.mean(np.abs(fallback_scaled)) / (np.mean(np.abs(base_episode_row)) + config.epsilon)
    )
    metrics.update(
        {
            "blpo/coverage/exact_inactive_role_active": (
                0.0
                if config.role_application == "local_residual"
                else float(np.mean(~exact_active & layer_active["role"]))
            ),
            "blpo/coverage/exact_role_inactive_stage_active": float(
                np.mean(~exact_active & ~layer_active["role"] & layer_active["stage"])
            ),
            "blpo/coverage/only_role_signal": (
                0.0
                if config.role_application == "local_residual"
                else float(
                    np.mean(
                        ~exact_active
                        & layer_active["role"]
                        & ~layer_active["stage"]
                    )
                )
            ),
            "blpo/coverage/both_exact_role_active": float(
                np.mean(exact_active & layer_active["role"])
            ),
            "blpo/coverage/only_stage_signal": float(
                np.mean(~exact_active & ~layer_active["role"] & layer_active["stage"])
            ),
            "blpo/conflict/exact_role": _conflict(exact, layer_adv["role"]),
            "blpo/conflict/exact_history_role": _conflict(
                exact, history_role
            ),
            "blpo/correlation/exact_role": _pearson(
                exact, layer_adv["role"], config.epsilon
            ),
            "blpo/correlation/exact_history_role": _pearson(
                exact, history_role, config.epsilon
            ),
            "blpo/conflict/role_stage": _conflict(layer_adv["role"], layer_adv["stage"]),
            "blpo/conflict/exact_stage": _conflict(exact, layer_adv["stage"]),
            "blpo/nan_inf_count": float(n - int(finite.sum())),
        }
    )
    metrics["blpo/advantage/micro_positive_ratio"] = float(np.mean(micro > 0))
    metrics["blpo/advantage/micro_negative_ratio"] = float(np.mean(micro < 0))
    metrics["blpo/advantage/micro_zero_ratio"] = float(np.mean(micro == 0))
    metrics["blpo/role/shareable_step_ratio"] = float(role_shareable.mean())
    if np.any(layer_active["role"] & ~role_shareable):
        raise RuntimeError("Role activated on a non-shareable physical step")

    if any(str(item) == "webshop" for item in key_batch.families):
        page_labels = np.asarray(
            [
                str(role[1])
                if str(family) == "webshop" and isinstance(role, tuple) and len(role) > 1
                else "non_webshop"
                for family, role in zip(key_batch.families, key_batch.role)
            ],
            dtype=object,
        )
        for page in sorted(set(str(item) for item in page_labels if str(item) != "non_webshop")):
            mask = page_labels == page
            prefix = f"blpo/webshop_stage/{page}"
            metrics[f"{prefix}/step_ratio"] = float(mask.mean())
            metrics[f"{prefix}/role_shareable_ratio"] = float(role_shareable[mask].mean())
            metrics[f"{prefix}/exact_active_ratio"] = float(exact_active[mask].mean())
            metrics[f"{prefix}/role_active_ratio"] = float(layer_active["role"][mask].mean())
            metrics[f"{prefix}/role_only_ratio"] = (
                0.0
                if config.role_application == "local_residual"
                else float(
                    np.mean(~exact_active[mask] & layer_active["role"][mask])
                )
            )
            if config.role_application == "local_residual":
                metrics[f"{prefix}/local_role_rescued_ratio"] = float(
                    np.mean(~exact_active[mask] & layer_active["role"][mask])
                )
            metrics[f"{prefix}/both_active_ratio"] = float(
                np.mean(exact_active[mask] & layer_active["role"][mask])
            )
            metrics[f"{prefix}/micro_mean"] = float(micro[mask].mean())
            metrics[f"{prefix}/micro_std"] = float(micro[mask].std())
            metrics[f"{prefix}/micro_nonzero_ratio"] = float(np.mean(np.abs(micro[mask]) > 0))
            metrics[f"{prefix}/role_advantage_mean"] = float(layer_adv["role"][mask].mean())
            metrics[f"{prefix}/role_advantage_std"] = float(layer_adv["role"][mask].std())
            metrics[f"{prefix}/exact_role_sign_conflict"] = _conflict(
                exact[mask], layer_adv["role"][mask]
            )
            metrics[f"{prefix}/exact_role_pearson"] = _pearson(
                exact[mask], layer_adv["role"][mask], config.epsilon
            )
            metrics[f"{prefix}/exact_history_role_sign_conflict"] = _conflict(
                exact[mask], history_role[mask]
            )
            metrics[f"{prefix}/exact_history_role_pearson"] = _pearson(
                exact[mask], history_role[mask], config.epsilon
            )
            metrics[f"{prefix}/history_role_active_ratio"] = float(
                history_role_active[mask].mean()
            )
            metrics[f"{prefix}/role_reliability_mean"] = float(
                role_reliability[mask].mean()
            )
            metrics[f"{prefix}/progress_prior_abs_mean"] = float(
                np.abs(progress_prior_component[mask]).mean()
            )

    if config.role_estimator == "residual_action":
        families = np.asarray(key_batch.families, dtype=object)
        for family in sorted(set(str(item) for item in key_batch.families)):
            mask = families == family
            display_family = FAMILY_METRIC_NAMES.get(family, family)
            prefix = f"blpo/residual_role/{display_family}"
            metrics[f"{prefix}/role_only_ratio"] = (
                0.0
                if config.role_application == "local_residual"
                else float(
                    np.mean(~exact_active[mask] & layer_active["role"][mask])
                )
            )
            if config.role_application == "local_residual":
                metrics[f"{prefix}/local_role_rescued_ratio"] = float(
                    np.mean(~exact_active[mask] & layer_active["role"][mask])
                )
            metrics[f"{prefix}/both_active_ratio"] = float(
                np.mean(exact_active[mask] & layer_active["role"][mask])
            )
            metrics[f"{prefix}/exact_active_ratio"] = float(
                exact_active[mask].mean()
            )
            metrics[f"{prefix}/exact_role_sign_conflict"] = _conflict(
                exact[mask], layer_adv["role"][mask]
            )
            metrics[f"{prefix}/exact_role_pearson"] = _pearson(
                exact[mask], layer_adv["role"][mask], config.epsilon
            )
            metrics[f"{prefix}/exact_history_role_sign_conflict"] = _conflict(
                exact[mask], history_role[mask]
            )
            metrics[f"{prefix}/exact_history_role_pearson"] = _pearson(
                exact[mask], history_role[mask], config.epsilon
            )
            metrics[f"{prefix}/history_role_active_ratio"] = float(
                history_role_active[mask].mean()
            )
            metrics[f"{prefix}/role_reliability_mean"] = float(
                role_reliability[mask].mean()
            )
            metrics[f"{prefix}/progress_prior_abs_mean"] = float(
                np.abs(progress_prior_component[mask]).mean()
            )

    if (
        config.assert_exact_alignment
        and config.layers == ("exact",)
        and not config.exact_action_balance
    ):
        reference, _ = returns.compute_reference_group_advantage(
            token_level_rewards=token_level_rewards,
            step_rewards=step_rewards,
            response_mask=response_mask,
            anchor_obs=batch.non_tensor_batch["anchor_obs"],
            index=uid,
            traj_index=traj_uid,
            epsilon=config.epsilon,
            step_advantage_w=config.step_advantage_w,
            mode=config.mode,
            enable_similarity=False,
        )
        original_groups = returns.build_reference_groups(
            batch.non_tensor_batch["anchor_obs"], uid
        )
        group_match = _partition(original_groups) == _partition(key_batch.exact)
        final_error = float(torch.max(torch.abs(scores - reference)).item())
        step_error = float(torch.max(torch.abs(micro_tokens - exact_tokens)).item())
        # The two equivalent paths accumulate float32 values in a different
        # order on GPU. Keep the alignment gate strict, but account for the
        # resulting scale-dependent roundoff instead of using a fixed 1e-6.
        alignment_scale = max(
            1.0,
            float(torch.max(torch.abs(reference)).item()),
            float(torch.max(torch.abs(exact_tokens)).item()),
        )
        dtype_epsilon = torch.finfo(reference.dtype).eps
        alignment_tolerance = max(
            4.0 * config.epsilon,
            8.0 * dtype_epsilon * alignment_scale,
        )
        metrics.update(
            {
                "blpo/alignment/group_partition_match": float(group_match),
                "blpo/alignment/group_size_match": float(
                    sorted(map(len, _partition(original_groups))) == sorted(map(len, _partition(key_batch.exact)))
                ),
                "blpo/alignment/step_advantage_max_error": step_error,
                "blpo/alignment/final_advantage_max_error": final_error,
                "blpo/alignment/numeric_tolerance": alignment_tolerance,
            }
        )
        if not group_match or step_error > alignment_tolerance or final_error > alignment_tolerance:
            raise RuntimeError(
                "Exact-only alignment failed: "
                f"group={group_match} step={step_error} final={final_error} "
                f"tolerance={alignment_tolerance}"
            )

    if config.step_records_dir is not None:
        rows_written = _write_step_records(
            output_dir=config.step_records_dir, run_seed=config.audit_run_seed,
            raw_sample_rate=config.step_records_raw_sample_rate, update=update,
            batch=batch, key_batch=key_batch, rewards=rewards, exact=exact,
            exact_active=exact_active, progress_raw=history_role,
            progress_final=layer_adv["role"], progress_active=layer_active["role"],
            role_shareable=role_shareable, role_actions=role_actions,
            role_audit=role_audit)
        metrics["blpo/audit/rows_written"] = float(rows_written)

    return scores, scores, metrics, key_batch.examples
