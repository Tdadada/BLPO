"""Deterministic local and task-progress keys for supported environments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .returns import to_hashable
from .environments.alfworld import (
    LIGHTS,
    TargetState,
    apply_text_action,
    component_done,
    component_targets_from_gamefile,
    target_progress_stage,
    target_relative_key,
)
from .environments.webshop import (
    goal_compact_role,
    normalized,
    public_exact_signature,
    task_fingerprint,
)


FAMILY_NAMES = {
    "pick_and_place_simple": "pick_and_place",
    "pick_two_obj_and_place": "pick_two_obj",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "look",
}


@dataclass(frozen=True)
class KeyBatch:
    exact: tuple[tuple[Any, ...], ...]
    exact_child: tuple[tuple[Any, ...], ...]
    role: tuple[tuple[Any, ...], ...]
    stage: tuple[tuple[Any, ...], ...]
    families: tuple[str, ...]
    examples: tuple[dict[str, Any], ...]


def task_family(gamefile: str) -> str:
    instance = Path(gamefile).parent.parent.name
    raw = instance.split("-", 1)[0]
    return FAMILY_NAMES.get(raw, raw)


def _target_instances(state: TargetState, object_type: str) -> set[str]:
    return {
        instance
        for instance, instance_type in state.object_instance_type.items()
        if instance_type == object_type
    }


def _goal_progress_descriptor(
    state: TargetState,
    component: Any,
) -> tuple[Any, ...]:
    """Keep only relations needed to identify the next unmet goal condition."""

    progress = target_progress_stage(state, component)
    obj = component.object_type
    rec = component.receptacle_type
    if component.kind == "place_count2":
        target_instances = _target_instances(state, obj)
        placed_instances = {
            instance
            for instance in target_instances
            if state.object_instance_role.get(instance) == f"in_recep:{rec}"
        }
        unplaced_seen = bool(
            (target_instances & state.seen_object_instances) - placed_instances
        )
        holding_unplaced = (
            bool(state.held_instance)
            and state.object_instance_type.get(state.held_instance) == obj
            and state.held_instance not in placed_instances
        )
        return (
            component.kind,
            progress,
            ("remaining_relation", holding_unplaced, unplaced_seen),
        )
    if component.kind == "examine_light":
        target_instances = _target_instances(state, obj)
        holding_target = (
            bool(state.held_instance)
            and state.object_instance_type.get(state.held_instance) == obj
        )
        under_lamp = bool(target_instances & state.under_light_instances)
        return (
            component.kind,
            progress,
            ("inspection_relation", holding_target, under_lamp),
        )
    return (component.kind, progress)


def _role_v1_descriptor(
    state: TargetState,
    component: Any,
    components: tuple[Any, ...],
) -> tuple[Any, ...]:
    extra: tuple[Any, ...] = ()
    if component.kind == "examine_light":
        extra = (
            "look_conditions",
            component.object_type in state.seen_object_types,
            bool(state.seen_receptacle_types & LIGHTS),
            bool(state.toggled_types & LIGHTS),
            state.last_receptacle_type in LIGHTS,
        )
    return (
        component.kind,
        target_relative_key(state, component, components, version="v2_sibling"),
        target_progress_stage(state, component),
        extra,
    )


def build_role_key(
    state: TargetState,
    gamefile: str,
    version: str = "goal_progress_v2",
) -> tuple[Any, ...]:
    """Build an entity-free Role key for the selected ALFWorld abstraction."""

    components = component_targets_from_gamefile(gamefile)
    family = task_family(gamefile)
    if version == "role_v1":
        descriptors = [
            _role_v1_descriptor(state, component, components)
            for component in components
        ]
        return (
            "role_v1",
            family,
            tuple(component.kind for component in components),
            tuple(descriptors),
            tuple(bool(component_done(state, component)) for component in components),
        )
    if version != "goal_progress_v2":
        raise ValueError(f"unsupported ALFWorld Role version: {version}")
    descriptors = [
        _goal_progress_descriptor(state, component)
        for component in components
    ]
    done_pattern = [bool(component_done(state, component)) for component in components]
    return (
        "goal_progress_role_v2",
        family,
        tuple(component.kind for component in components),
        tuple(descriptors),
        tuple(done_pattern),
    )


def _current_stage(role_key: tuple[Any, ...]) -> str:
    _, family, _, descriptors, done_pattern = role_key
    if all(done_pattern):
        return "complete"

    current = next(i for i, done in enumerate(done_pattern) if not done)
    kind, progress, *extra = descriptors[current]
    bucket = int(progress[1]) if len(progress) > 1 else 0

    if kind == "examine_light":
        _, _, object_seen, light_seen, light_on, near_light = progress
        holding_target, under_lamp = extra[0][1:]
        if not object_seen:
            return "navigation"
        if not holding_target:
            return "acquire"
        if not (near_light or light_seen):
            return "navigation_light"
        if not light_on and not under_lamp:
            return "operate"
        return "inspect"
    if kind in {"hot", "cool", "clean"}:
        if bucket <= 0:
            return "navigation"
        if bucket == 1:
            return "acquire"
        if bucket == 2:
            return "operate"
        return "transform"
    if kind in {"place", "place_count2"}:
        if bucket <= 0:
            return "navigation"
        if bucket == 1:
            return "acquire"
        return "place"
    return family


def project_stage_key(role_key: tuple[Any, ...]) -> tuple[Any, ...]:
    """Project Role to a coarser key; no rollout state is consulted here."""

    if role_key[0] == "role_v1":
        _, family, target_kinds, descriptors, done_pattern = role_key
        progress_buckets = tuple(int(item[2][1]) for item in descriptors)
        if all(done_pattern):
            current_stage = "complete"
        else:
            current = next(
                index for index, done in enumerate(done_pattern) if not done
            )
            kind = target_kinds[current]
            bucket = progress_buckets[current]
            if bucket <= 0:
                current_stage = "navigation"
            elif kind in {"hot", "cool", "clean"}:
                current_stage = ("acquire", "operate", "transform")[
                    min(bucket, 3) - 1
                ]
            elif kind in {"place", "place_count2"}:
                current_stage = "acquire" if bucket == 1 else "place"
            elif kind == "examine_light":
                current_stage = ("acquire", "navigation_light", "inspect")[
                    min(bucket, 3) - 1
                ]
            else:
                current_stage = family
        return (
            "stage_v1",
            family,
            target_kinds,
            progress_buckets,
            done_pattern,
            current_stage,
        )

    _, family, target_kinds, descriptors, done_pattern = role_key
    progress_buckets = tuple(
        int(item[1][1]) if len(item[1]) > 1 else 0
        for item in descriptors
    )
    return (
        "goal_progress_stage_v2",
        family,
        target_kinds,
        progress_buckets,
        done_pattern,
        _current_stage(role_key),
    )


def _build_webshop_key_batch(
    batch: Any,
    webshop_abstraction: str,
) -> KeyBatch:
    n = len(batch)
    traj_uids = batch.non_tensor_batch["traj_uid"]
    group_uids = batch.non_tensor_batch["uid"]
    observations = batch.non_tensor_batch["anchor_obs"]
    actions = batch.non_tensor_batch.get("text_action", [""] * n)
    tasks = batch.non_tensor_batch["webshop_task"]
    public_states = batch.non_tensor_batch["webshop_pre_public_state"]

    exact = []
    exact_child = []
    role = []
    stage = []
    examples = []
    for idx in range(n):
        task = str(tasks[idx])
        public_state = public_states[idx]
        if not task or not isinstance(public_state, dict):
            raise RuntimeError(f"WebShop BLPO metadata missing at row {idx}")
        obs_key = to_hashable(observations[idx])
        signature = public_exact_signature(public_state)
        role_key = goal_compact_role(
            task,
            public_state,
            str(observations[idx]),
            abstraction_version=webshop_abstraction,
        )
        # Local groups use (prompt group, anchor observation). Public UI
        # metadata is reserved for the cross-state task-progress abstraction.
        exact.append((str(group_uids[idx]), obs_key))
        exact_child.append(("webshop", task_fingerprint(task), obs_key))
        role.append(role_key)
        stage.append(("webshop_stage_v1", role_key[1]))
        if len(examples) < 20:
            examples.append(
                {
                    "family": "webshop",
                    "gamefile": "",
                    "trajectory": str(traj_uids[idx]),
                    "action": str(actions[idx]),
                    "task_fingerprint": task_fingerprint(task),
                    "task_preview": normalized(task)[:240],
                    "exact_observation": str(observations[idx]),
                    "public_exact_signature": signature,
                    "role_key": role_key,
                    "stage_key": stage[-1],
                }
            )
    return KeyBatch(
        exact=tuple(exact),
        exact_child=tuple(exact_child),
        role=tuple(role),
        stage=tuple(stage),
        families=("webshop",) * n,
        examples=tuple(examples),
    )


def build_key_batch(
    batch: Any,
    webshop_abstraction: str = "goal_compact_v1",
    alfworld_role_version: str = "goal_progress_v2",
) -> KeyBatch:
    n = len(batch)
    if (
        "webshop_task" in batch.non_tensor_batch
        and "webshop_pre_public_state" in batch.non_tensor_batch
    ):
        return _build_webshop_key_batch(batch, webshop_abstraction)

    traj_uids = batch.non_tensor_batch["traj_uid"]
    group_uids = batch.non_tensor_batch["uid"]
    observations = batch.non_tensor_batch["anchor_obs"]
    gamefiles = batch.non_tensor_batch.get("gamefile", [""] * n)
    actions = batch.non_tensor_batch.get("text_action", [""] * n)
    # The -0.1 format penalty is format-only. Rule-state updates must
    # still follow what ALFWorld could actually execute, otherwise an
    # inadmissible action would create fictional Role/Stage progress.
    valid = batch.non_tensor_batch.get(
        "is_action_admissible",
        batch.non_tensor_batch.get("is_action_valid", np.ones(n, dtype=bool)),
    )

    by_traj: dict[str, list[int]] = defaultdict(list)
    for idx, traj_uid in enumerate(traj_uids):
        by_traj[str(traj_uid)].append(idx)

    exact: list[tuple[Any, ...] | None] = [None] * n
    exact_child: list[tuple[Any, ...] | None] = [None] * n
    role: list[tuple[Any, ...] | None] = [None] * n
    stage: list[tuple[Any, ...] | None] = [None] * n
    families: list[str | None] = [None] * n
    examples: list[dict[str, Any]] = []
    example_counts: dict[str, int] = defaultdict(int)

    for traj_uid, indices in by_traj.items():
        first = indices[0]
        gamefile = str(gamefiles[first])
        if not gamefile:
            raise RuntimeError(f"BLPO key generation requires gamefile for trajectory {traj_uid}")
        family = task_family(gamefile)
        state = TargetState()
        for idx in indices:
            obs_key = to_hashable(observations[idx])
            role_key = build_role_key(state, gamefile, alfworld_role_version)
            stage_key = project_stage_key(role_key)
            # Exact keeps the original prompt UID boundary. The stable child key
            # omits random UUIDs and is used only for EMA support diagnostics.
            exact[idx] = (str(group_uids[idx]), obs_key)
            exact_child[idx] = (family, obs_key)
            role[idx] = role_key
            stage[idx] = stage_key
            families[idx] = family
            if example_counts[family] < 20:
                examples.append(
                    {
                        "family": family,
                        "gamefile": gamefile,
                        "trajectory": traj_uid,
                        "action": str(actions[idx]),
                        "exact_observation": str(observations[idx]),
                        "role_key": role_key,
                        "stage_key": stage_key,
                    }
                )
                example_counts[family] += 1
            state = apply_text_action(state, str(actions[idx]), valid=bool(valid[idx]))

    if any(item is None for item in exact + exact_child + role + stage + families):
        raise RuntimeError("BLPO key generation left unassigned rows")
    return KeyBatch(
        exact=tuple(exact),
        exact_child=tuple(exact_child),
        role=tuple(role),
        stage=tuple(stage),
        families=tuple(families),
        examples=tuple(examples),
    )
