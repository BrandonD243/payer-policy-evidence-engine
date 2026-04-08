# evaluators/pa_required_evaluator.py

import os
import yaml
from typing import Dict, Any, List


# ===============================
# Helpers
# ===============================

def _normalize(value):
    """Normalize strings for safe comparison."""
    if value is None:
        return None
    return str(value).strip().lower()


def _list_normalize(values):
    """Normalize list of values."""
    if not values:
        return []
    return [_normalize(v) for v in values]


def _matches_field(rule_values: List[str], context_value: str) -> bool:
    """
    Field match logic.

    Rules:
    - If rule field missing/empty → wildcard (match all)
    - Otherwise context must be in rule list
    """
    if not rule_values:
        return True

    if context_value is None:
        return False

    return _normalize(context_value) in _list_normalize(rule_values)


# ===============================
# Rule Loader
# ===============================

def load_pa_rules(payer_name: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rules_path = os.path.join(base_dir, "payers", payer_name, "pa_rules.yaml")

    if not os.path.exists(rules_path):
        return [], False

    with open(rules_path, "r") as f:
        data = yaml.safe_load(f) or {}

    rules = data.get("pa_rules", [])
    default_requires_pa = data.get("default_requires_pa", False)

    # sort by priority (highest first)
    rules_sorted = sorted(
        rules,
        key=lambda r: r.get("priority", 0),
        reverse=True
    )

    return rules_sorted, default_requires_pa


# ===============================
# Core Evaluator
# ===============================

def check_pa_required(
    payer_name: str,
    cpt_code: str,
    context: Dict[str, Any] | None = None
) -> bool:
    """
    Determine if prior authorization is required.

    Context may include:
      - state
      - plan_type
      - place_of_service
    """

    context = context or {}

    rules, default_requires_pa = load_pa_rules(payer_name)
    print("PA RULES LOADED:", rules)

    cpt_code_norm = _normalize(cpt_code)
    state = _normalize(context.get("state"))
    plan_type = _normalize(context.get("plan_type"))
    pos = _normalize(context.get("place_of_service"))

    for rule in rules:
        match = rule.get("match", {})

        rule_cpts = _list_normalize(match.get("cpt_codes"))
        rule_states = match.get("states")
        rule_plan_types = match.get("plan_types")
        rule_pos = match.get("place_of_service")

        # --- CPT must match (required anchor) ---
        if rule_cpts and cpt_code_norm not in rule_cpts:
            continue

        # --- optional dimensions ---
        if not _matches_field(rule_states, state):
            continue

        if not _matches_field(rule_plan_types, plan_type):
            continue

        if not _matches_field(rule_pos, pos):
            continue

        return rule.get("requires_pa", True)

    return default_requires_pa