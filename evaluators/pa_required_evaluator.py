# evaluators/pa_required_evaluator.py

import os
import yaml


def load_pa_rules(payer_name: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rules_path = os.path.join(base_dir, "payers", payer_name, "pa_rules.yaml")

    if not os.path.exists(rules_path):
        return []

    with open(rules_path, "r") as f:
        return yaml.safe_load(f).get("pa_rules", [])


def check_pa_required(payer_name: str, cpt_code: str) -> bool:
    rules = load_pa_rules(payer_name)

    for rule in rules:
        if cpt_code in rule.get("cpt_codes", []):
            return rule.get("requires_pa", True)

    return False