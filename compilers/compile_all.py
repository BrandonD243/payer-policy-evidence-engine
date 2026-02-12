# compilers/compile_all.py
import yaml
import json
from pathlib import Path
import argparse
from extractors.openai_concept_extractor import extract_concepts_from_text  # updated extractor name


def compile_concepts(concept_registry_path: str, output_path: str) -> dict:
    with open(concept_registry_path, "r") as f:
        concept_data = json.load(f) if concept_registry_path.endswith(".json") else yaml.safe_load(f)
    concepts_kg = {"concepts": concept_data.get("concepts", {})}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(concepts_kg, f, indent=2)
    print(f"[Concept Compiler] Compiled {len(concepts_kg['concepts'])} concepts to {output_path}")
    return concepts_kg


def compile_payer_rules(payers_dir: str, output_path: str) -> dict:
    payers_kg = {}
    for payer_path in Path(payers_dir).glob("*/72148/"):
        payer_name = payer_path.parent.name
        cpt_code = payer_path.name
        payers_kg.setdefault(payer_name, {})[cpt_code] = {}
        for filename in ["policy.yaml", "approval_clauses.yaml", "exclusion_clauses.yaml"]:
            file_path = payer_path / filename
            if file_path.exists():
                with open(file_path, "r") as f:
                    key = filename.replace(".yaml", "")
                    payers_kg[payer_name][cpt_code][key] = yaml.safe_load(f)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payers_kg, f, indent=2)
    print(f"[Payer Compiler] Compiled payer rules for {len(payers_kg)} payers to {output_path}")
    return payers_kg


def link_concepts_to_payers(concepts_kg: dict, payers_kg: dict, user_docs_dir: str, output_path: str) -> dict:
    # Extract concepts from the user documents using your existing extractor
    user_concepts_list = extract_concepts_from_text(user_docs_dir, concepts_kg)

    # Convert list to dict keyed by concept ID
    user_concepts = {c["id"]: c for c in user_concepts_list}

    merged_kg = {}
    for payer_name, cpt_data in payers_kg.items():
        merged_kg[payer_name] = {}
        for cpt_code, rules in cpt_data.items():
            merged_kg[payer_name][cpt_code] = {"resolved_concepts": {}, "clauses": rules}
            for concept_id, concept_info in user_concepts.items():
                if concept_id in concepts_kg.get("concepts", {}):
                    merged_kg[payer_name][cpt_code]["resolved_concepts"][concept_id] = concept_info

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged_kg, f, indent=2)

    print(f"[Linker] Merged user concepts with payers into {output_path}")
    return merged_kg


def main():
    parser = argparse.ArgumentParser(description="Compile concepts, payer rules, and merge with user documents")
    parser.add_argument("--concepts", type=str, required=True, help="Path to concept_registry.yaml")
    parser.add_argument("--payers", type=str, required=True, help="Path to payers directory")
    parser.add_argument("--user_docs", type=str, required=True, help="Path to user documents directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write compiled knowledge graphs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    concepts_kg = compile_concepts(args.concepts, output_dir / "concepts_kg.json")
    payers_kg = compile_payer_rules(args.payers, output_dir / "payers_kg.json")
    merged_kg = link_concepts_to_payers(concepts_kg, payers_kg, args.user_docs, output_dir / "merged_kg.json")

    print("\n[Compile All] Knowledge graph compilation complete!")


if __name__ == "__main__":
    main()
