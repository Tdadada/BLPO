"""Public-state Exact, Goal-Compact Role, and action keys for WebShop."""

from __future__ import annotations

import hashlib
import re
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "below",
    "buy",
    "dollar",
    "dollars",
    "find",
    "for",
    "from",
    "in",
    "is",
    "it",
    "lower",
    "me",
    "of",
    "on",
    "or",
    "price",
    "than",
    "that",
    "the",
    "this",
    "to",
    "under",
    "with",
}
CONTROL_ACTIONS = {
    "back to search": "BACK_TO_SEARCH",
    "< prev": "PREVIOUS",
    "prev": "PREVIOUS",
    "previous": "PREVIOUS",
    "next >": "NEXT_RESULTS",
    "next": "NEXT_RESULTS",
    "description": "INSPECT_DESCRIPTION",
    "features": "INSPECT_FEATURES",
    "reviews": "INSPECT_REVIEWS",
    "buy now": "BUY",
}
OPTION_PATTERN = re.compile(
    r"(?:\bwith|,\s*and)\s+([a-z][a-z0-9 /&_-]{0,30}?):\s*"
    r"(.+?)(?=,\s*and\s+[a-z][a-z0-9 /&_-]{0,30}?:|,\s*and\s+price\b|[.]?$)",
    re.IGNORECASE,
)
PRICE_LIMIT_PATTERN = re.compile(
    r"price\s+lower\s+than\s+\$?([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"^\s*(search|click)\[(.*)\]\s*$", re.IGNORECASE)
PRICE_PATTERN = re.compile(
    r"(?:Price:\s*)?\$([0-9]+(?:\.[0-9]+)?)(?:\s+to\s+\$([0-9]+(?:\.[0-9]+)?))?",
    re.IGNORECASE,
)
ABSTRACTION_VERSIONS = {
    "goal_compact_v1",
    "constraint_progress_v2",
    "constraint_progress_v3",
    "decision_frontier_v4",
}
UNSHAREABLE_PROGRESS_ACTIONS = {
    "FORMAT_INVALID",
    "ENV_INVALID",
    "UNRESOLVED_ACTION",
}


def normalized(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9#]+", " ", str(value).lower()),
    ).strip()


def content_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalized(value).split()
        if len(token) > 1 and token not in STOPWORDS and not token.isdigit()
    }


def _bucket_count(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return "1_2"
    if value <= 5:
        return "3_5"
    return "6_plus"


def _coverage_bucket(covered: int, total: int) -> str:
    if total <= 0:
        return "none_required"
    if covered <= 0:
        return "none"
    if covered >= total:
        return "all"
    return "high" if covered / total >= 2.0 / 3.0 else "low"


def parse_goal(task: str) -> dict[str, Any]:
    options: dict[str, str] = {}
    for match in OPTION_PATTERN.finditer(task):
        key = normalized(match.group(1)).rsplit(" with ", 1)[-1]
        value = normalized(match.group(2))
        if key and value:
            options[key] = value
    price_match = PRICE_LIMIT_PATTERN.search(task)
    product_text = OPTION_PATTERN.sub(" ", task)
    product_text = PRICE_LIMIT_PATTERN.sub(" ", product_text)
    return {
        "options": options,
        "price_limit": float(price_match.group(1)) if price_match else None,
        "tokens": content_tokens(product_text),
    }


def _query_progress(
    query: Any,
    goal: dict[str, Any],
) -> tuple[str, str, int]:
    query_tokens = content_tokens(" ".join(query) if isinstance(query, list) else query)
    covered = len(query_tokens & goal["tokens"])
    option_covered = sum(
        bool(content_tokens(value))
        and content_tokens(value).issubset(query_tokens)
        for value in goal["options"].values()
    )
    return (
        _coverage_bucket(covered, len(goal["tokens"])),
        _coverage_bucket(option_covered, len(goal["options"])),
        covered,
    )


def _split_segments(anchor: str) -> list[str]:
    return [
        str(segment).strip().strip("'\"").strip()
        for segment in str(anchor).split("[SEP]")
        if str(segment).strip().strip("'\"").strip()
    ]


def _parse_action(text: str) -> tuple[str, str]:
    match = ACTION_PATTERN.match(str(text))
    if not match:
        return "", ""
    return match.group(1).lower(), match.group(2).strip().lower()


def _option_matches(selected: str, desired: str) -> bool:
    left = normalized(selected)
    right = normalized(desired)
    return bool(left and right and (left == right or left in right or right in left))


def _page_price(anchor: str, price_limit: float | None) -> str:
    if price_limit is None:
        return "no_limit"
    match = PRICE_PATTERN.search(str(anchor))
    if not match:
        return "unknown"
    low = float(match.group(1))
    high = float(match.group(2) or match.group(1))
    if high <= price_limit:
        return "ok"
    if low > price_limit:
        return "high"
    return "mixed"


def _product_title(anchor: str) -> str:
    segments = _split_segments(anchor)
    for index, segment in enumerate(segments):
        if PRICE_PATTERN.search(segment) and index > 0:
            return segments[index - 1]
    return ""


def _result_candidates(
    public_state: dict[str, Any],
    anchor: str,
    goal: dict[str, Any],
) -> list[dict[str, Any]]:
    segments = _split_segments(anchor)
    lowered = [segment.lower() for segment in segments]
    controls = set(CONTROL_ACTIONS)
    candidates = []
    for clickable in public_state.get("available_actions", {}).get("clickables", []):
        key = normalized(clickable)
        if key in controls:
            continue
        try:
            index = lowered.index(str(clickable).lower())
        except ValueError:
            continue
        title = segments[index + 1] if index + 1 < len(segments) else ""
        price_text = segments[index + 2] if index + 2 < len(segments) else ""
        overlap = len(content_tokens(title) & goal["tokens"])
        candidates.append(
            {
                "action_arg": str(clickable).lower(),
                "overlap": overlap,
                "coverage": _coverage_bucket(overlap, len(goal["tokens"])),
                "price": _page_price(price_text, goal["price_limit"]),
            }
        )
    return candidates


def _result_progress(
    task: str,
    public_state: dict[str, Any],
    anchor: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    goal = parse_goal(task)
    candidates = _result_candidates(public_state, anchor, goal)
    best_overlap = max((item["overlap"] for item in candidates), default=0)
    affordable = [
        item
        for item in candidates
        if item["price"] in {"ok", "no_limit", "mixed"}
    ]
    best_affordable = max((item["overlap"] for item in affordable), default=0)
    query_coverage, query_options, _ = _query_progress(
        public_state.get("query") or [],
        goal,
    )
    return (
        {
            "goal": goal,
            "query_coverage": query_coverage,
            "query_options": query_options,
            "best_coverage": _coverage_bucket(best_overlap, len(goal["tokens"])),
            "best_affordable_coverage": _coverage_bucket(
                best_affordable,
                len(goal["tokens"]),
            ),
        },
        candidates,
    )


def _candidate_quality(candidate: dict[str, Any]) -> str:
    """Quality of opening one visible candidate under hard constraints."""

    if candidate["price"] == "high" or candidate["coverage"] == "none":
        return "weak"
    if (
        candidate["coverage"] in {"high", "all"}
        and candidate["price"] in {"ok", "no_limit"}
    ):
        return "strong"
    return "partial"


def _best_actionable_candidate(candidates: list[dict[str, Any]]) -> str:
    qualities = {_candidate_quality(candidate) for candidate in candidates}
    if "strong" in qualities:
        return "strong"
    if "partial" in qualities:
        return "partial"
    return "none"


def _selected_progress(
    public_state: dict[str, Any],
    goal: dict[str, Any],
) -> tuple[int, int, int]:
    selected = {
        normalized(key): normalized(value)
        for key, value in (public_state.get("selected_options") or {}).items()
    }
    correct = 0
    wrong = 0
    for key, value in selected.items():
        desired = goal["options"].get(key)
        if desired is None:
            wrong += 1
        elif _option_matches(value, desired):
            correct += 1
        else:
            wrong += 1
    return correct, wrong, len(goal["options"])


def _option_availability(
    public_state: dict[str, Any],
    goal: dict[str, Any],
) -> str:
    desired = list(goal["options"].values())
    if not desired:
        return "none_required"
    clickables = [
        normalized(item)
        for item in public_state.get("available_actions", {}).get("clickables", [])
    ]
    found = sum(
        any(_option_matches(clickable, value) for clickable in clickables)
        for value in desired
    )
    if found == 0:
        return "none"
    if found == len(desired):
        return "all"
    return "partial"


def public_exact_signature(public_state: dict[str, Any]) -> tuple[Any, ...]:
    """Observable Markov state omitted from some rendered WebShop pages."""

    selected = tuple(
        sorted(
            (normalized(key), normalized(value))
            for key, value in (public_state.get("selected_options") or {}).items()
        )
    )
    return (
        "webshop_public_exact_v1",
        str(public_state.get("page_type") or "unknown"),
        str(public_state.get("subpage_type") or ""),
        tuple(normalized(item) for item in public_state.get("query") or []),
        int(public_state.get("page_index") or 1),
        normalized(public_state.get("current_asin") or ""),
        selected,
    )


def task_fingerprint(task: str) -> str:
    return hashlib.sha1(normalized(task).encode("utf-8")).hexdigest()


def _goal_compact_role_v1(
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
) -> tuple[Any, ...]:
    """Relational WebShop state with concrete products and option values removed."""

    goal = parse_goal(task)
    page = str(public_state.get("page_type") or "unknown")
    action_counts = public_state.get("action_counts") or {}
    correct, wrong, required = _selected_progress(public_state, goal)
    if required == 0:
        selection = "none_required"
    elif correct >= required and wrong == 0:
        selection = "all"
    elif correct > 0:
        selection = "partial"
    else:
        selection = "none"

    if page == "search":
        state = ("retry" if int(action_counts.get("search", 0) or 0) else "first",)
    elif page == "search_results":
        query = " ".join(str(item) for item in public_state.get("query") or [])
        state = (
            "later_page" if int(public_state.get("page_index") or 1) > 1 else "first_page",
            _bucket_count(len(content_tokens(query) & goal["tokens"])),
        )
    elif page in {"item_page", "item_sub_page"}:
        title_overlap = len(content_tokens(_product_title(anchor_obs)) & goal["tokens"])
        inspected = sum(
            int(action_counts.get(key, 0) or 0)
            for key in ("description", "features", "reviews")
        )
        state = (
            _bucket_count(title_overlap),
            _page_price(anchor_obs, goal["price_limit"]),
            _option_availability(public_state, goal),
            selection,
            bool(wrong),
            "inspected" if inspected else "uninspected",
            str(public_state.get("subpage_type") or ""),
        )
    else:
        state = ()
    return ("webshop_goal_compact_v1", page, *state)


def _item_progress(
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
) -> tuple[str, str]:
    goal = parse_goal(task)
    title = _product_title(anchor_obs)
    evidence = title or anchor_obs
    overlap = len(content_tokens(evidence) & goal["tokens"])
    coverage = _coverage_bucket(overlap, len(goal["tokens"]))
    price = _page_price(anchor_obs, goal["price_limit"])
    correct, wrong, required = _selected_progress(public_state, goal)

    if wrong or price == "high" or coverage == "none":
        item_status = "mismatch"
    elif coverage in {"high", "all"} and price in {"ok", "no_limit"}:
        item_status = "match"
    else:
        item_status = "uncertain"
    option_ready = (
        "yes"
        if (required == 0 or (correct >= required and wrong == 0))
        else "no"
    )
    return item_status, option_ready


def constraint_progress_role(
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
) -> tuple[Any, ...]:
    """Compact pre-action goal progress and next-stage readiness."""

    page = str(public_state.get("page_type") or "unknown")
    if page == "search":
        # WebShop submits a complete query in one action. The proposed query
        # belongs to the action abstraction and must not leak into this key.
        state = ("entry",)
    elif page == "search_results":
        _, candidates = _result_progress(task, public_state, anchor_obs)
        state = (_best_actionable_candidate(candidates),)
    elif page in {"item_page", "item_sub_page"}:
        state = _item_progress(task, public_state, anchor_obs)
    else:
        state = ("unknown",)
    return ("webshop_constraint_progress_v2", page, *state)


def constraint_progress_role_v3(
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
) -> tuple[Any, ...]:
    """Stage-local readiness without product, page, or query identity."""

    page = str(public_state.get("page_type") or "unknown")
    if page == "search":
        state = ("entry",)
    elif page == "search_results":
        progress, candidates = _result_progress(task, public_state, anchor_obs)
        query_ready = (
            progress["query_coverage"] in {"high", "all"}
            and progress["query_options"] in {"all", "none_required"}
        )
        product_clickables = [
            item
            for item in public_state.get("available_actions", {}).get(
                "clickables", []
            )
            if normalized(item) not in CONTROL_ACTIONS
        ]
        candidate_state = (
            "unknown"
            if product_clickables and not candidates
            else _best_actionable_candidate(candidates)
        )
        state = (
            "query_ready" if query_ready else "query_not_ready",
            candidate_state,
        )
    elif page in {"item_page", "item_sub_page"}:
        state = _item_progress(task, public_state, anchor_obs)
    else:
        state = ("unknown",)
    return ("webshop_constraint_progress_v3", page, *state)


def _result_frontier(
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
) -> str:
    """Return the one reliably observable decision bottleneck on Results."""

    progress, candidates = _result_progress(task, public_state, anchor_obs)
    goal = progress["goal"]
    if not goal["tokens"]:
        return "bottom"

    clickables = public_state.get("available_actions", {}).get("clickables", [])
    product_clickables = [
        item for item in clickables if normalized(item) not in CONTROL_ACTIONS
    ]
    # Product links exist but their titles/prices could not be grounded in the
    # rendered observation. Sharing this state would silently mix unrelated
    # candidates.
    if product_clickables and not candidates:
        return "bottom"

    if any(_candidate_quality(candidate) == "strong" for candidate in candidates):
        return "candidate_ready"

    # Option values are intentionally excluded here. They are normally chosen
    # on the item page; requiring them in the query caused the v3 Cartesian key
    # to label useful searches as deficient.
    if progress["query_coverage"] in {"none", "low"}:
        return "query_bottleneck"

    controls = {CONTROL_ACTIONS.get(normalized(item)) for item in clickables}
    if "NEXT_RESULTS" in controls:
        return "results_bottleneck"
    return "bottom"


def _item_frontier(
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
) -> str:
    """Return the active unmet condition on an actionable item page."""

    if str(public_state.get("page_type") or "unknown") != "item_page":
        # Item subpages generally offer only a return action. The useful
        # decision was the preceding INSPECT_ITEM step, so do not create a
        # cross-state Role group here.
        return "bottom"

    goal = parse_goal(task)
    if not goal["tokens"]:
        return "bottom"

    title = _product_title(anchor_obs)
    evidence = title or anchor_obs
    overlap = len(content_tokens(evidence) & goal["tokens"])
    coverage = _coverage_bucket(overlap, len(goal["tokens"]))
    price = _page_price(anchor_obs, goal["price_limit"])
    correct, wrong, required = _selected_progress(public_state, goal)
    availability = _option_availability(public_state, goal)

    if price == "high" or coverage == "none":
        return "item_reject"
    if required > 0 and (correct < required or wrong):
        if availability in {"all", "partial"}:
            return "option_needed"
        return "item_reject"
    if coverage in {"high", "all"} and price in {"ok", "no_limit"}:
        return "purchase_ready"

    action_counts = public_state.get("action_counts") or {}
    inspected = any(
        int(action_counts.get(key, 0) or 0) > 0
        for key in ("description", "features", "reviews")
    )
    if not inspected:
        return "information_needed"
    return "bottom"


def decision_frontier_role(
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
) -> tuple[Any, ...]:
    """Decision-frontier Role with an explicit non-shareable bottom."""

    page = str(public_state.get("page_type") or "unknown")
    if page == "search":
        frontier = "bottom"
    elif page == "search_results":
        frontier = _result_frontier(task, public_state, anchor_obs)
    elif page in {"item_page", "item_sub_page"}:
        frontier = _item_frontier(task, public_state, anchor_obs)
    else:
        frontier = "bottom"
    return ("webshop_decision_frontier_v4", page, frontier)


def goal_compact_role(
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
    *,
    abstraction_version: str = "goal_compact_v1",
) -> tuple[Any, ...]:
    if abstraction_version not in ABSTRACTION_VERSIONS:
        raise ValueError(f"unknown WebShop abstraction: {abstraction_version}")
    if abstraction_version == "constraint_progress_v2":
        return constraint_progress_role(task, public_state, anchor_obs)
    if abstraction_version == "constraint_progress_v3":
        return constraint_progress_role_v3(task, public_state, anchor_obs)
    if abstraction_version == "decision_frontier_v4":
        return decision_frontier_role(task, public_state, anchor_obs)
    return _goal_compact_role_v1(task, public_state, anchor_obs)


def _candidate_rank(
    action_arg: str,
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
) -> str:
    candidates = [
        str(item)
        for item in public_state.get("available_actions", {}).get("clickables", [])
        if normalized(item) not in CONTROL_ACTIONS
    ]
    candidate_lookup = {item.lower(): item for item in candidates}
    if action_arg not in candidate_lookup:
        return "unknown"
    segments = _split_segments(anchor_obs)
    lowered_segments = [segment.lower() for segment in segments]
    goal_tokens = parse_goal(task)["tokens"]
    scores: list[tuple[str, int]] = []
    for candidate in candidates:
        try:
            index = lowered_segments.index(candidate.lower())
        except ValueError:
            scores.append((candidate.lower(), 0))
            continue
        title = segments[index + 1] if index + 1 < len(segments) else ""
        scores.append((candidate.lower(), len(content_tokens(title) & goal_tokens)))
    ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
    rank = next(
        (index + 1 for index, item in enumerate(ranked) if item[0] == action_arg),
        len(ranked),
    )
    return "top1" if rank == 1 else ("top3" if rank <= 3 else "other")


def _goal_compact_action_v1(
    action: str,
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
    *,
    format_valid: bool,
    admissible: bool,
) -> str:
    if not format_valid:
        return "FORMAT_INVALID"
    if not admissible:
        return "ENV_INVALID"

    name, arg = _parse_action(action)
    goal = parse_goal(task)
    page = str(public_state.get("page_type") or "unknown")
    if name == "search":
        arg_tokens = content_tokens(arg)
        overlap = len(arg_tokens & goal["tokens"])
        option_value = any(
            content_tokens(value) <= arg_tokens
            for value in goal["options"].values()
            if content_tokens(value)
        )
        current_query = normalized(" ".join(public_state.get("query") or []))
        repeated = bool(current_query and normalized(arg) == current_query)
        return (
            f"SEARCH_{_bucket_count(overlap)}"
            f"_OPTION_{int(option_value)}_REPEAT_{int(repeated)}"
        )
    if name != "click":
        return "OTHER"

    control = CONTROL_ACTIONS.get(normalized(arg))
    if control == "BUY":
        correct, wrong, required = _selected_progress(public_state, goal)
        price = _page_price(anchor_obs, goal["price_limit"])
        if price == "high":
            return "BUY_PRICE_HIGH"
        if required > 0 and (correct < required or wrong):
            return "BUY_MISSING_OR_WRONG_OPTION"
        return "BUY_READY"
    if control:
        return control
    if page == "search_results":
        return f"OPEN_PRODUCT_{_candidate_rank(arg, task, public_state, anchor_obs)}"
    if page in {"item_page", "item_sub_page"}:
        selected = {
            normalized(value)
            for value in (public_state.get("selected_options") or {}).values()
        }
        if normalized(arg) in selected:
            return "RESELECT_SAME_VALUE"
        if any(_option_matches(arg, value) for value in goal["options"].values()):
            return "SELECT_REQUIRED_VALUE"
        return "SELECT_OTHER_VALUE"
    return "OTHER"


def constraint_progress_action(
    action: str,
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
    *,
    format_valid: bool,
    admissible: bool,
) -> str:
    """Compact action relation to the current stage's unmet constraints."""

    if not format_valid:
        return "FORMAT_INVALID"
    if not admissible:
        return "ENV_INVALID"

    name, arg = _parse_action(action)
    goal = parse_goal(task)
    page = str(public_state.get("page_type") or "unknown")
    if name == "search":
        current = _query_progress(public_state.get("query") or [], goal)
        proposed = _query_progress(arg, goal)
        repeated = normalized(arg) == normalized(
            " ".join(public_state.get("query") or [])
        )
        ready = (
            proposed[0] in {"high", "all"}
            and proposed[1] in {"all", "none_required"}
        )
        degraded = repeated or proposed[2] <= 0 or proposed[2] < current[2]
        if page == "search_results":
            return (
                "REPEAT_OR_DEGRADE_QUERY"
                if degraded or not ready
                else "REFINE_QUERY"
            )
        if page in {"item_page", "item_sub_page"}:
            return "BACK" if ready else "UNRESOLVED_ACTION"
        if degraded:
            return "QUERY_DEGRADED_OR_REPEAT"
        return "QUERY_GOOD_COVERAGE" if ready else "QUERY_PARTIAL_COVERAGE"
    if name != "click":
        return "UNRESOLVED_ACTION"

    control = CONTROL_ACTIONS.get(normalized(arg))
    if page == "search_results":
        _, candidates = _result_progress(task, public_state, anchor_obs)
        if control in {"NEXT_RESULTS", "PREVIOUS"}:
            return "CONTINUE_RESULTS"
        if control == "BACK_TO_SEARCH":
            return "REFINE_QUERY"
        candidate = next(
            (item for item in candidates if item["action_arg"] == arg),
            None,
        )
        if candidate is None:
            return "UNRESOLVED_ACTION"
        return f"OPEN_{_candidate_quality(candidate).upper()}"

    if page not in {"item_page", "item_sub_page"}:
        return "UNRESOLVED_ACTION"

    if control == "BUY":
        return "BUY"
    if control in {"INSPECT_DESCRIPTION", "INSPECT_FEATURES", "INSPECT_REVIEWS"}:
        return "INSPECT"
    if control in {"BACK_TO_SEARCH", "PREVIOUS"}:
        return "BACK"
    if control:
        return "UNRESOLVED_ACTION"

    selected = {
        normalized(value)
        for value in (public_state.get("selected_options") or {}).values()
    }
    if normalized(arg) in selected:
        return "SELECT_OTHER"
    if any(_option_matches(arg, value) for value in goal["options"].values()):
        return "SELECT_REQUIRED"
    return "SELECT_OTHER"


def decision_frontier_action(
    action: str,
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
    *,
    format_valid: bool,
    admissible: bool,
) -> str:
    """Map an action to its direction relative to the active frontier."""

    if not format_valid:
        return "FORMAT_INVALID"
    if not admissible:
        return "ENV_INVALID"

    name, arg = _parse_action(action)
    page = str(public_state.get("page_type") or "unknown")
    goal = parse_goal(task)

    # Query alternatives stay inside Exact. Search has no reliable cross-state
    # Role comparison because all admissible actions belong to one continuous
    # query family.
    if page == "search":
        return "UNRESOLVED_ACTION"

    if page == "search_results":
        if name == "search":
            current = _query_progress(public_state.get("query") or [], goal)
            proposed = _query_progress(arg, goal)
            repeated = normalized(arg) == normalized(
                " ".join(public_state.get("query") or [])
            )
            if not repeated and proposed[2] > current[2]:
                return "BACKTRACK_QUERY"
            return "UNRESOLVED_ACTION"
        if name != "click":
            return "UNRESOLVED_ACTION"

        control = CONTROL_ACTIONS.get(normalized(arg))
        if control == "BACK_TO_SEARCH":
            return "BACKTRACK_QUERY"
        if control == "NEXT_RESULTS":
            return "EXPLORE_RESULTS"
        if control:
            # PREVIOUS_RESULTS is deliberately not shared: without reliable
            # page memory it can be either recovery or a redundant revisit.
            return "UNRESOLVED_ACTION"

        _, candidates = _result_progress(task, public_state, anchor_obs)
        candidate = next(
            (item for item in candidates if item["action_arg"] == arg),
            None,
        )
        if candidate is None:
            return "UNRESOLVED_ACTION"
        quality = _candidate_quality(candidate)
        if quality == "strong":
            return "OPEN_FRONTIER"
        if quality == "weak" or any(
            _candidate_quality(item) == "strong" for item in candidates
        ):
            return "OPEN_NONFRONTIER"
        return "UNRESOLVED_ACTION"

    if page != "item_page":
        return "UNRESOLVED_ACTION"
    if name != "click":
        return "UNRESOLVED_ACTION"

    control = CONTROL_ACTIONS.get(normalized(arg))
    if control == "BUY":
        return "BUY"
    if control in {"INSPECT_DESCRIPTION", "INSPECT_FEATURES", "INSPECT_REVIEWS"}:
        return "INSPECT_ITEM"
    if control in {"BACK_TO_SEARCH", "PREVIOUS"}:
        return "BACKTRACK_ITEM"
    if control:
        return "UNRESOLVED_ACTION"

    selected = {
        normalized(key): normalized(value)
        for key, value in (public_state.get("selected_options") or {}).items()
    }
    clicked = normalized(arg)
    for key, desired in goal["options"].items():
        if not _option_matches(clicked, desired):
            continue
        if _option_matches(selected.get(key, ""), desired):
            return "VIOLATE_OR_REPEAT_OPTION"
        return "SATISFY_OPTION"
    return "VIOLATE_OR_REPEAT_OPTION"


def goal_compact_action(
    action: str,
    task: str,
    public_state: dict[str, Any],
    anchor_obs: str,
    *,
    format_valid: bool,
    admissible: bool,
    abstraction_version: str = "goal_compact_v1",
) -> str:
    if abstraction_version not in ABSTRACTION_VERSIONS:
        raise ValueError(f"unknown WebShop abstraction: {abstraction_version}")
    if abstraction_version in {"constraint_progress_v2", "constraint_progress_v3"}:
        result = constraint_progress_action(
            action,
            task,
            public_state,
            anchor_obs,
            format_valid=format_valid,
            admissible=admissible,
        )
        if abstraction_version == "constraint_progress_v3":
            if result == "CONTINUE_RESULTS":
                _, arg = _parse_action(action)
                control = CONTROL_ACTIONS.get(normalized(arg))
                if control == "NEXT_RESULTS":
                    return "NEXT_RESULTS"
                if control == "PREVIOUS":
                    return "PREVIOUS_RESULTS"
        return result
    if abstraction_version == "decision_frontier_v4":
        return decision_frontier_action(
            action,
            task,
            public_state,
            anchor_obs,
            format_valid=format_valid,
            admissible=admissible,
        )
    return _goal_compact_action_v1(
        action,
        task,
        public_state,
        anchor_obs,
        format_valid=format_valid,
        admissible=admissible,
    )


def progress_action_is_shareable(action: str) -> bool:
    return bool(action and action not in UNSHAREABLE_PROGRESS_ACTIONS)


def progress_role_is_shareable(role: tuple[Any, ...]) -> bool:
    """Reject cross-state sharing when a target relation failed to parse."""

    values = {str(item) for item in role[2:]}
    return not values.intersection({"unknown", "bottom"})
