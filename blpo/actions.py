"""Concrete-to-abstract action projections used by Role estimators."""

from __future__ import annotations

import re
import unicodedata

from .environments.alfworld import (
    LIGHTS,
    component_targets_from_gamefile,
    normalize_name,
    strip_instance_id,
)
from .environments.webshop import goal_compact_action


def canonical_exact_action(
    action: str,
    *,
    format_valid: bool = True,
    admissible: bool = True,
) -> tuple[str, str]:
    """Canonicalize a concrete action without erasing entity arguments.

    Valid, inadmissible, and malformed commands remain separate so the
    baseline invalid-action penalty cannot be averaged into a valid command.
    """

    text = unicodedata.normalize("NFKC", str(action)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not format_valid:
        validity = "format_invalid"
    elif not admissible:
        validity = "inadmissible"
    else:
        validity = "valid"
    return validity, text


def _command_parts(action: str) -> tuple[str, str, str]:
    text = normalize_name(action)
    patterns = (
        ("take", r"take (.+?) from (.+)"),
        ("put", r"(?:put|move) (.+?) (?:in|on|to) (.+)"),
        ("go", r"go to (.+)"),
        ("open", r"open (.+)"),
        ("close", r"close (.+)"),
        ("toggle", r"(?:toggle|turn on) (.+)"),
        ("use", r"use (.+?)(?: with (.+))?$"),
        ("transform", r"(?:heat|cool|clean) (.+?)(?: with (.+))?$"),
        ("inspect", r"(?:look at|examine) (.+)"),
    )
    for verb, pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            groups = match.groups()
            second = groups[1] if len(groups) > 1 and groups[1] else ""
            return verb, groups[0], second
    return "", "", ""


def abstract_action(
    action: str,
    gamefile: str,
    *,
    webshop_abstraction: str = "goal_compact_v1",
    webshop_task: str = "",
    webshop_public_state=None,
    anchor_obs: str = "",
    format_valid: bool = True,
    admissible: bool = True,
) -> str:
    """Map a concrete command to the task-progress action vocabulary."""

    if webshop_task and webshop_public_state is not None:
        return goal_compact_action(
            action,
            webshop_task,
            webshop_public_state,
            anchor_obs,
            format_valid=format_valid,
            admissible=admissible,
            abstraction_version=webshop_abstraction,
        )

    components = component_targets_from_gamefile(gamefile)
    primary = components[0]
    destination = next(
        (
            component.receptacle_type
            for component in components
            if component.kind in {"place", "place_count2"}
        ),
        primary.receptacle_type,
    )
    transform_tools = {
        "hot": {"microwave"},
        "cool": {"fridge"},
        "clean": {"sinkbasin", "bathtubbasin"},
    }.get(primary.kind, set())
    lamp_types = set(LIGHTS)

    text = normalize_name(action)
    verb, first, second = _command_parts(text)
    first_type = strip_instance_id(first)
    second_type = strip_instance_id(second)
    if primary.kind == "examine_light" and verb == "use":
        if first_type in lamp_types and second_type == primary.object_type:
            return "INSPECT_TARGET"
        if first_type in lamp_types:
            return "ACTIVATE_LAMP"
    if primary.kind == "examine_light" and verb == "inspect":
        return (
            "EXAMINE_TARGET"
            if first_type == primary.object_type
            else "OBSERVE_OTHER"
        )
    if verb == "go":
        if first_type in lamp_types:
            return "NAVIGATE_LAMP"
        if first_type in transform_tools:
            return "NAVIGATE_TOOL"
        if first_type == destination:
            return "NAVIGATE_DESTINATION"
        return "SEARCH_ENV"
    if verb == "open":
        if first_type == destination:
            return "OPEN_DESTINATION"
        if first_type in transform_tools:
            return "OPEN_TOOL"
        return "OPEN_SOURCE"
    if verb == "take" and first_type == primary.object_type:
        if primary.kind == "place_count2":
            return (
                "RETAKE_PLACED_TARGET"
                if second_type == destination
                else "TAKE_UNPLACED_TARGET"
            )
        return "TAKE_TARGET"
    if verb == "put" and first_type == primary.object_type:
        if second_type == destination:
            return "PUT_TARGET_DESTINATION"
        if second_type in transform_tools:
            return "PUT_TARGET_TOOL"
        return "PUT_TARGET_OTHER"
    if verb == "transform" and first_type == primary.object_type:
        return "TRANSFORM_TARGET"
    if verb in {"toggle", "use"} and (
        first_type in lamp_types
        or second_type in lamp_types
        or first_type in transform_tools
        or second_type in transform_tools
    ):
        return "ACTIVATE_TOOL"
    if verb == "inspect":
        return "OBSERVE_OTHER"
    if text.startswith("look"):
        return "OBSERVE_ENV"
    if text.startswith("inventory"):
        return "INVENTORY"
    if verb in {"go", "open", "close"}:
        return "SEARCH"
    return "OTHER"
