"""Rule-based ALFWorld target extraction and target-relative abstractions.

This module has two state backends:

* facts backend: consumes TextWorld facts from ``infos["facts"]`` when the
  environment requests them.
* action backend: a lightweight tracker for gold walkthroughs and offline
  probes.  It is also useful as a fallback when facts are unavailable.

The first training integration should prefer facts, because facts reflect the
actual environment state after invalid or unusual model actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


HEATERS = frozenset({"microwave"})
COOLERS = frozenset({"fridge"})
CLEANERS = frozenset({"sinkbasin"})
LIGHTS = frozenset({"desklamp", "floorlamp"})


def normalize_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text == "none":
        return ""
    return text


def strip_instance_id(value: Any) -> str:
    """Return an object/receptacle type from names like ``mug 2``."""

    text = normalize_name(value)
    text = re.sub(r"\s+\d+$", "", text)
    text = text.replace(" ", "")
    return text


@dataclass(frozen=True, order=True)
class TargetComponent:
    """A single symbolic ALFWorld target component."""

    kind: str
    object_type: str
    receptacle_type: str = ""

    @property
    def type_key(self) -> tuple[str]:
        return (self.kind,)

    @property
    def exact_key(self) -> tuple[str, str, str]:
        return (self.kind, self.object_type, self.receptacle_type)


@dataclass
class TargetState:
    """Compact state used by target-relative abstractions."""

    held_type: str = ""
    object_role: dict[str, str] = field(default_factory=dict)
    seen_object_types: set[str] = field(default_factory=set)
    hot: set[str] = field(default_factory=set)
    cool: set[str] = field(default_factory=set)
    clean: set[str] = field(default_factory=set)
    examined: set[str] = field(default_factory=set)
    placed_count: dict[tuple[str, str], int] = field(default_factory=dict)
    last_receptacle_type: str = ""
    seen_receptacle_types: set[str] = field(default_factory=set)
    open_receptacles: set[str] = field(default_factory=set)
    toggled_types: set[str] = field(default_factory=set)
    # Instance-level fields are intentionally additive.  Legacy target-credit
    # modes only consume the type-level fields above; target_segment_td uses
    # these to distinguish the two same-type objects in PickTwo and to retain
    # the actual object/light relation for LOOK.
    held_instance: str = ""
    object_instance_type: dict[str, str] = field(default_factory=dict)
    object_instance_role: dict[str, str] = field(default_factory=dict)
    seen_object_instances: set[str] = field(default_factory=set)
    under_light_instances: set[str] = field(default_factory=set)

    def copy(self) -> "TargetState":
        return TargetState(
            held_type=self.held_type,
            object_role=dict(self.object_role),
            seen_object_types=set(self.seen_object_types),
            hot=set(self.hot),
            cool=set(self.cool),
            clean=set(self.clean),
            examined=set(self.examined),
            placed_count=dict(self.placed_count),
            last_receptacle_type=self.last_receptacle_type,
            seen_receptacle_types=set(self.seen_receptacle_types),
            open_receptacles=set(self.open_receptacles),
            toggled_types=set(self.toggled_types),
            held_instance=self.held_instance,
            object_instance_type=dict(self.object_instance_type),
            object_instance_role=dict(self.object_instance_role),
            seen_object_instances=set(self.seen_object_instances),
            under_light_instances=set(self.under_light_instances),
        )


def instance_key(value: Any) -> str:
    """Return a stable action-level instance identifier without type folding."""

    return re.sub(r"\s+", "", normalize_name(value))


def component_targets_from_task(
    task_type: str,
    object_target: str,
    parent_target: str = "",
    toggle_target: str = "",
) -> tuple[TargetComponent, ...]:
    """Factor an ALFWorld task into target components."""

    task = normalize_name(task_type)
    obj = strip_instance_id(object_target)
    parent = strip_instance_id(parent_target)
    toggle = strip_instance_id(toggle_target)

    if task == "pick_and_place_simple":
        return (TargetComponent("place", obj, parent),)
    if task == "pick_heat_then_place_in_recep":
        return (TargetComponent("hot", obj), TargetComponent("place", obj, parent))
    if task == "pick_cool_then_place_in_recep":
        return (TargetComponent("cool", obj), TargetComponent("place", obj, parent))
    if task == "pick_clean_then_place_in_recep":
        return (TargetComponent("clean", obj), TargetComponent("place", obj, parent))
    if task == "pick_two_obj_and_place":
        return (TargetComponent("place_count2", obj, parent),)
    if task == "look_at_obj_in_light":
        return (TargetComponent("examine_light", obj, toggle or parent),)
    return (TargetComponent(task, obj, parent),)


def component_targets_from_gamefile(gamefile: str | Path) -> tuple[TargetComponent, ...]:
    """Extract target components from an ALFWorld ``game.tw-pddl`` path."""

    inst = Path(gamefile).parent.parent.name
    parts = inst.split("-")
    task = parts[0] if parts else ""
    obj = parts[1] if len(parts) > 1 else ""
    mrecep_or_toggle = parts[2] if len(parts) > 2 else ""
    parent = parts[3] if len(parts) > 3 else ""
    toggle = parent if task == "look_at_obj_in_light" else mrecep_or_toggle
    return component_targets_from_task(task, obj, parent, toggle)


def component_done(state: TargetState, component: TargetComponent) -> bool:
    obj = component.object_type
    rec = component.receptacle_type
    if component.kind == "hot":
        return obj in state.hot
    if component.kind == "cool":
        return obj in state.cool
    if component.kind == "clean":
        return obj in state.clean
    if component.kind == "place":
        return state.object_role.get(obj) == f"in_recep:{rec}"
    if component.kind == "place_count2":
        return state.placed_count.get((obj, rec), 0) >= 2
    if component.kind == "examine_light":
        return obj in state.examined
    return False


def _simple_role(role: str, target_receptacle: str) -> str:
    if role == "held":
        return "held"
    if target_receptacle and role == f"in_recep:{target_receptacle}":
        return "at_target_recep"
    if role.startswith("in_recep:"):
        return "in_other_recep"
    if role.startswith("in_") or "_at:" in role:
        return role.split(":", 1)[0]
    return "unknown"


def sibling_pattern(
    state: TargetState,
    component: TargetComponent,
    all_components: Sequence[TargetComponent],
) -> tuple[tuple[str, bool], ...]:
    """Return sibling target completion for bound multi-factor tasks."""

    return tuple(
        sorted(
            (other.kind, component_done(state, other))
            for other in all_components
            if other != component and other.object_type == component.object_type
        )
    )


def target_progress_stage(state: TargetState, component: TargetComponent) -> tuple[Any, ...]:
    """Return task-agnostic milestone features for target-relative progress.

    The stage is intentionally coarse. It gives the value table a way to share
    prerequisite progress without binding to concrete object ids, while leaving
    final success semantics in ``component_done`` unchanged.
    """

    obj = component.object_type
    rec = component.receptacle_type
    role = state.object_role.get(obj, "unknown")
    held = state.held_type == obj
    obj_seen = obj in state.seen_object_types or role != "unknown" or held
    rec_seen = bool(rec) and rec in state.seen_receptacle_types

    if component.kind in {"hot", "cool", "clean"}:
        if component.kind == "hot":
            tool_set = HEATERS
            done = obj in state.hot
            at_tool = role == "in_heater" or role.startswith("heated_at:")
        elif component.kind == "cool":
            tool_set = COOLERS
            done = obj in state.cool
            at_tool = role == "in_cooler" or role.startswith("cooled_at:")
        else:
            tool_set = CLEANERS
            done = obj in state.clean
            at_tool = role == "in_cleaner" or role.startswith("cleaned_at:")
        tool_seen = bool(state.seen_receptacle_types & tool_set)
        tool_access = state.last_receptacle_type in tool_set or bool(state.open_receptacles & tool_set)
        if done:
            stage = 4
        elif at_tool:
            stage = 3
        elif held and (tool_seen or tool_access):
            stage = 2
        elif held or obj_seen:
            stage = 1
        else:
            stage = 0
        return ("progress", stage, obj_seen, tool_seen, tool_access)

    if component.kind == "place":
        at_target = role == f"in_recep:{rec}"
        near_target = state.last_receptacle_type == rec
        if at_target:
            stage = 3
        elif held and (near_target or rec_seen):
            stage = 2
        elif held or obj_seen:
            stage = 1
        else:
            stage = 0
        return ("progress", stage, obj_seen, rec_seen, near_target)

    if component.kind == "place_count2":
        count = min(state.placed_count.get((obj, rec), 0), 2)
        near_target = state.last_receptacle_type == rec
        need_more = count < 2
        if count >= 2:
            stage = 4
        elif count == 1 and held:
            stage = 3
        elif count == 1:
            stage = 2
        elif held or obj_seen:
            stage = 1
        else:
            stage = 0
        return ("progress", stage, count, need_more, held, rec_seen, near_target)

    if component.kind == "examine_light":
        light_on = bool(state.toggled_types & LIGHTS)
        light_seen = bool(state.seen_receptacle_types & LIGHTS)
        near_light = state.last_receptacle_type in LIGHTS
        done = obj in state.examined
        if done:
            stage = 4
        elif obj_seen and light_on:
            stage = 3
        elif obj_seen and (near_light or light_seen):
            stage = 2
        elif obj_seen or light_seen:
            stage = 1
        else:
            stage = 0
        return ("progress", stage, obj_seen, light_seen, light_on, near_light)

    return ("progress", int(component_done(state, component)))


def target_relative_key(
    state: TargetState,
    component: TargetComponent,
    all_components: Sequence[TargetComponent] = (),
    version: str = "v2_sibling",
) -> tuple[Any, ...]:
    """Compute a target-relative state key.

    ``v2_sibling`` is the original implementation: v2 features plus a sibling
    completion pattern only when the task has interacting components.
    ``v3_progress`` appends coarse prerequisite milestones to improve credit for
    multi-step targets such as look-at and place-two.
    """

    obj = component.object_type
    rec = component.receptacle_type
    role = state.object_role.get(obj, "unknown")
    done = component_done(state, component)
    held = state.held_type == obj
    simple_role = _simple_role(role, rec)

    if version == "v1":
        base = (component.kind, done, simple_role, held)
    elif component.kind == "hot":
        in_tool = role == "in_heater" or role.startswith("heated_at:")
        tool_access = state.last_receptacle_type in HEATERS or bool(state.open_receptacles & HEATERS)
        base = (component.kind, done, simple_role, held, in_tool, tool_access)
    elif component.kind == "cool":
        in_tool = role == "in_cooler" or role.startswith("cooled_at:")
        tool_access = state.last_receptacle_type in COOLERS or bool(state.open_receptacles & COOLERS)
        base = (component.kind, done, simple_role, held, in_tool, tool_access)
    elif component.kind == "clean":
        in_tool = role == "in_cleaner" or role.startswith("cleaned_at:")
        tool_access = state.last_receptacle_type in CLEANERS or bool(state.open_receptacles & CLEANERS)
        base = (component.kind, done, simple_role, held, in_tool, tool_access)
    elif component.kind in ("place", "place_count2"):
        at_target = role == f"in_recep:{rec}"
        near_target = state.last_receptacle_type == rec
        count_bucket = min(state.placed_count.get((obj, rec), 0), 2) if component.kind == "place_count2" else int(at_target)
        base = (component.kind, done, simple_role, held, at_target, near_target, count_bucket)
    elif component.kind == "examine_light":
        light_on = bool(state.toggled_types & LIGHTS)
        near_light = state.last_receptacle_type in LIGHTS
        base = (component.kind, done, simple_role, held, light_on, near_light)
    else:
        base = (component.kind, done, simple_role, held)

    if version in ("v2", "v1"):
        return base
    if version in ("v3", "v2_sibling"):
        pattern = sibling_pattern(state, component, all_components)
        return base + ((pattern if pattern else ()),)
    if version in ("v3_progress", "v3_milestone"):
        pattern = sibling_pattern(state, component, all_components)
        progress = target_progress_stage(state, component)
        return base + ((pattern if pattern else ()), progress)
    raise ValueError(f"unknown target-relative key version: {version}")


def apply_text_action(state: TargetState, action: str, valid: bool = True) -> TargetState:
    """Update a lightweight state tracker from an ALFWorld text action."""

    next_state = state.copy()
    if not valid:
        return next_state
    text = normalize_name(action)

    if text.startswith("go to "):
        rec_type = strip_instance_id(text[len("go to ") :])
        next_state.last_receptacle_type = rec_type
        next_state.seen_receptacle_types.add(rec_type)
        return next_state

    match = re.match(r"open (.+)", text)
    if match:
        rec_type = strip_instance_id(match.group(1))
        next_state.open_receptacles.add(rec_type)
        next_state.seen_receptacle_types.add(rec_type)
        return next_state

    match = re.match(r"close (.+)", text)
    if match:
        rec_type = strip_instance_id(match.group(1))
        next_state.open_receptacles.discard(rec_type)
        next_state.seen_receptacle_types.add(rec_type)
        return next_state

    match = re.match(r"take (.+?) from (.+)", text)
    if match:
        obj_instance = instance_key(match.group(1))
        obj_type = strip_instance_id(match.group(1))
        rec_type = strip_instance_id(match.group(2))
        previous_role = next_state.object_instance_role.get(obj_instance, "")
        if previous_role == f"in_recep:{rec_type}":
            count_key = (obj_type, rec_type)
            next_state.placed_count[count_key] = max(
                0,
                next_state.placed_count.get(count_key, 0) - 1,
            )
        next_state.held_type = obj_type
        next_state.held_instance = obj_instance
        next_state.seen_object_types.add(obj_type)
        next_state.seen_object_instances.add(obj_instance)
        next_state.object_instance_type[obj_instance] = obj_type
        next_state.object_instance_role[obj_instance] = "held"
        next_state.seen_receptacle_types.add(rec_type)
        next_state.object_role[obj_type] = "held"
        next_state.last_receptacle_type = rec_type
        return next_state

    match = re.match(r"(?:put|move) (.+?) (?:in|on|to) (.+)", text)
    if match:
        obj_instance = instance_key(match.group(1))
        obj_type = strip_instance_id(match.group(1))
        rec_type = strip_instance_id(match.group(2))
        previous_role = next_state.object_instance_role.get(obj_instance, "")
        next_state.held_type = ""
        if next_state.held_instance == obj_instance or not next_state.held_instance:
            next_state.held_instance = ""
        next_state.seen_object_types.add(obj_type)
        next_state.seen_object_instances.add(obj_instance)
        next_state.object_instance_type[obj_instance] = obj_type
        next_state.seen_receptacle_types.add(rec_type)
        if rec_type in HEATERS:
            role = "in_heater"
        elif rec_type in COOLERS:
            role = "in_cooler"
        elif rec_type in CLEANERS:
            role = "in_cleaner"
        else:
            role = f"in_recep:{rec_type}"
        next_state.object_role[obj_type] = role
        next_state.object_instance_role[obj_instance] = role
        next_state.last_receptacle_type = rec_type
        if previous_role != role:
            count_key = (obj_type, rec_type)
            next_state.placed_count[count_key] = (
                next_state.placed_count.get(count_key, 0) + 1
            )
        return next_state

    for verb, attr, suffix in (
        ("heat", next_state.hot, "heated_at"),
        ("cool", next_state.cool, "cooled_at"),
        ("clean", next_state.clean, "cleaned_at"),
    ):
        match = re.match(rf"{verb} (.+?) with (.+)", text)
        if match:
            obj_instance = instance_key(match.group(1))
            obj_type = strip_instance_id(match.group(1))
            rec_type = strip_instance_id(match.group(2))
            attr.add(obj_type)
            next_state.seen_object_types.add(obj_type)
            next_state.seen_object_instances.add(obj_instance)
            next_state.object_instance_type[obj_instance] = obj_type
            next_state.object_instance_role[obj_instance] = f"{suffix}:{rec_type}"
            next_state.seen_receptacle_types.add(rec_type)
            next_state.object_role[obj_type] = f"{suffix}:{rec_type}"
            next_state.last_receptacle_type = rec_type
            return next_state

    match = re.match(r"use (.+?) with (.+)", text)
    if match:
        first_instance, second_instance = instance_key(match.group(1)), instance_key(match.group(2))
        first_type, second_type = strip_instance_id(match.group(1)), strip_instance_id(match.group(2))
        # ALFWorld's successful light command is ``use lamp with object``.
        # Retain the legacy fallback for other ``use`` commands.
        if first_type in LIGHTS:
            light_type, obj_type, obj_instance = first_type, second_type, second_instance
            next_state.toggled_types.add(light_type)
            next_state.seen_receptacle_types.add(light_type)
            next_state.seen_object_types.add(obj_type)
            next_state.seen_object_instances.add(obj_instance)
            next_state.object_instance_type[obj_instance] = obj_type
            next_state.under_light_instances.add(obj_instance)
            next_state.examined.add(obj_type)
            next_state.last_receptacle_type = light_type
        else:
            obj_type, rec_type = first_type, second_type
            next_state.seen_object_types.add(obj_type)
            next_state.seen_object_instances.add(first_instance)
            next_state.object_instance_type[first_instance] = obj_type
            next_state.seen_receptacle_types.add(rec_type)
            next_state.examined.add(obj_type)
            next_state.toggled_types.add(rec_type)
            next_state.last_receptacle_type = rec_type
        return next_state

    match = re.match(r"(?:toggle|turn on) (.+)", text)
    if match:
        rec_type = strip_instance_id(match.group(1))
        next_state.toggled_types.add(rec_type)
        next_state.seen_receptacle_types.add(rec_type)
    return next_state


def _fact_name(fact: Any) -> str:
    return normalize_name(getattr(fact, "name", None) or (fact[0] if isinstance(fact, (list, tuple)) and fact else ""))


def _fact_args(fact: Any) -> list[str]:
    if hasattr(fact, "arguments"):
        args = getattr(fact, "arguments")
    elif isinstance(fact, (list, tuple)) and len(fact) > 1:
        args = fact[1:]
    else:
        text = str(fact)
        match = re.match(r"([A-Za-z_]+)\((.*)\)", text)
        if not match:
            return []
        args = [part.strip() for part in match.group(2).split(",")]
    result = []
    for arg in args:
        name = getattr(arg, "name", arg)
        result.append(normalize_name(name))
    return result


def state_from_facts(facts: Iterable[Any]) -> TargetState:
    """Build ``TargetState`` from TextWorld facts.

    The parser is deliberately permissive because TextWorld proposition objects
    and serialized facts can differ slightly across wrappers.
    """

    state = TargetState()
    object_types: dict[str, str] = {}
    receptacle_types: dict[str, str] = {}
    held_objects: set[str] = set()
    in_recep: dict[str, str] = {}

    for fact in facts:
        name = _fact_name(fact)
        args = _fact_args(fact)
        if not args:
            continue
        if name in {"objecttype", "object_type"} and len(args) >= 2:
            object_types[args[0]] = strip_instance_id(args[1])
        elif name in {"receptacletype", "receptacle_type"} and len(args) >= 2:
            receptacle_types[args[0]] = strip_instance_id(args[1])
        elif name in {"holds", "inventory"}:
            held_objects.add(args[-1])
        elif name == "inreceptacle" and len(args) >= 2:
            in_recep[args[0]] = args[1]
        elif name == "ishot":
            state.hot.add(strip_instance_id(object_types.get(args[0], args[0])))
        elif name == "iscool":
            state.cool.add(strip_instance_id(object_types.get(args[0], args[0])))
        elif name == "isclean":
            state.clean.add(strip_instance_id(object_types.get(args[0], args[0])))
        elif name == "istoggled":
            state.toggled_types.add(strip_instance_id(receptacle_types.get(args[0], args[0])))
        elif name == "isopen":
            state.open_receptacles.add(strip_instance_id(receptacle_types.get(args[0], args[0])))

    for obj in held_objects:
        obj_type = strip_instance_id(object_types.get(obj, obj))
        obj_instance = instance_key(obj)
        state.held_type = obj_type
        state.held_instance = obj_instance
        state.seen_object_types.add(obj_type)
        state.seen_object_instances.add(obj_instance)
        state.object_instance_type[obj_instance] = obj_type
        state.object_instance_role[obj_instance] = "held"
        state.object_role[obj_type] = "held"

    for obj, rec in in_recep.items():
        obj_type = strip_instance_id(object_types.get(obj, obj))
        obj_instance = instance_key(obj)
        rec_type = strip_instance_id(receptacle_types.get(rec, rec))
        state.seen_object_types.add(obj_type)
        state.seen_receptacle_types.add(rec_type)
        if rec_type in HEATERS:
            role = "in_heater"
        elif rec_type in COOLERS:
            role = "in_cooler"
        elif rec_type in CLEANERS:
            role = "in_cleaner"
        else:
            role = f"in_recep:{rec_type}"
        if state.object_role.get(obj_type) != "held":
            state.object_role[obj_type] = role
        if state.object_instance_role.get(obj_instance) != "held":
            state.object_instance_role[obj_instance] = role
        state.object_instance_type[obj_instance] = obj_type
        state.seen_object_instances.add(obj_instance)
    return state
