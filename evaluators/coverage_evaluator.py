# evaluators/coverage_evaluator.py

import os
import yaml
from typing import Dict, Any


def load_coverage_rules(payer_name: str) -> Dict[str, Any]:
    """
    Load coverage rules for a payer.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rules_path = os.path.join(base_dir, "payers", payer_name, "coverage_rules.yaml")

    if not os.path.exists(rules_path):
        return {"coverage_rules": [], "default_coverage_status": "covered"}

    with open(rules_path, "r") as f:
        return yaml.safe_load(f) or {}


def _matches(rule_match: Dict[str, Any], cpt_code: str, context: Dict[str, Any]) -> bool:
    """
    Generic matcher for coverage rules.
    Easily extensible.
    """

    # --- CPT match ---
    cpt_list = rule_match.get("cpt_codes")
    if cpt_list and cpt_code not in cpt_list:
        return False

    # --- State match ---
    states = rule_match.get("states")
    if states:
        if context.get("state") not in states:
            return False

    # --- Plan type match (future-safe) ---
    plan_types = rule_match.get("plan_types")
    if plan_types:
        if context.get("plan_type") not in plan_types:
            return False

    return True


def check_coverage(
    payer_name: str,
    cpt_code: str,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Determine coverage status for a CPT code.

    Returns:
        {
            "coverage_status": "covered | non_covered | invalid | expired",
            "rule_id": str | None
        }
    """

    context = context or {}
    rules_data = load_coverage_rules(payer_name)

    rules = rules_data.get("coverage_rules", [])
    default_status = rules_data.get("default_coverage_status", "covered")

    # Highest priority wins
    sorted_rules = sorted(
        rules,
        key=lambda r: r.get("priority", 0),
        reverse=True,
    )

    for rule in sorted_rules:
        match_block = rule.get("match", {})
        if _matches(match_block, cpt_code, context):
            return {
                "coverage_status": rule.get("coverage_status", "covered"),
                "rule_id": rule.get("rule_id"),
            }

    # fallback
    return {
        "coverage_status": default_status,
        "rule_id": None,
    }