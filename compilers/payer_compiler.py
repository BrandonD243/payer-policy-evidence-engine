import yaml
import json
from pathlib import Path


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compile_payers(payers_root: str, output_dir: str):
    payers_kg = {}

    for payer_dir in Path(payers_root).iterdir():
        if not payer_dir.is_dir():
            continue

        payer_name = payer_dir.name
        payers_kg[payer_name] = {}

        for cpt_dir in payer_dir.iterdir():
            if not cpt_dir.is_dir():
                continue

            cpt_code = cpt_dir.name

            policy = load_yaml(cpt_dir / "policy.yaml")
            approvals = load_yaml(cpt_dir / "approval_clauses.yaml")
            exclusions = load_yaml(cpt_dir / "exclusion_clauses.yaml")
            recommendations = load_yaml(cpt_dir / "recommendation_registry.yaml")

            payers_kg[payer_name][cpt_code] = {
                "policy": policy,
                "clauses": {
                    "approval": approvals.get("approval_clauses", []),
                    "exclusion": exclusions.get("exclusion_clauses", []),
                },
                "recommendations": recommendations.get("recommendations", {}),
                "decision_logic": policy.get("decision_logic", {}),
            }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    yaml_path = Path(output_dir) / "payers_kg.yaml"
    json_path = Path(output_dir) / "payers_kg.json"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(payers_kg, f)

    with open(json_path, "w") as f:
        json.dump(payers_kg, f, indent=2)

    print(f"[PayerCompiler] Wrote {yaml_path} and {json_path}")

    return payers_kg
