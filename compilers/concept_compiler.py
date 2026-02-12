import yaml
import json
from pathlib import Path


def compile_concepts(concept_registry_path: str, output_dir: str):
    with open(concept_registry_path, "r") as f:
        registry = yaml.safe_load(f)

    concepts_kg = {"concepts": {}}

    for concept_id, concept_data in registry.get("concepts", {}).items():
        concepts_kg["concepts"][concept_id] = concept_data

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    yaml_path = Path(output_dir) / "concepts_kg.yaml"
    json_path = Path(output_dir) / "concepts_kg.json"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(concepts_kg, f)

    with open(json_path, "w") as f:
        json.dump(concepts_kg, f, indent=2)

    print(f"[ConceptCompiler] Wrote {yaml_path} and {json_path}")

    return concepts_kg
