# compilers/concept_payer_linker.py
import yaml
from pathlib import Path
from typing import Dict, List

def link_concepts_to_payers(concepts_kg: Dict, payers_kg: Dict, user_concepts: Dict, output_path: str):
    """
    user_concepts: Dict[concept_id -> {"certainty": str, "confidence": str}]
    """
    merged_kg = {}

    for payer_name, cpt_data in payers_kg.items():
        merged_kg[payer_name] = {}
        for cpt_code, rules in cpt_data.items():
            merged_kg[payer_name][cpt_code] = {"resolved_concepts": {}, "clauses": rules}

            # Only include concepts that exist in user input
            for concept_id, concept_info in user_concepts.items():
                if concept_id in concepts_kg.get("concepts", {}):
                    merged_kg[payer_name][cpt_code]["resolved_concepts"][concept_id] = concept_info

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(merged_kg, f)

    print(f"[Linker] Merged user concepts with payers into {output_path}")
    return merged_kg

if __name__ == "__main__":
    # Example usage with dummy user concepts
    with open("compiled_kg/concepts_kg.yaml") as f:
        concepts_kg = yaml.safe_load(f)
    with open("compiled_kg/payers_kg.yaml") as f:
        payers_kg = yaml.safe_load(f)

    user_concepts = {
        "spinal_stenosis": {"certainty": "confirmed", "confidence": "strong"},
        "radiculopathy": {"certainty": "confirmed", "confidence": "moderate"},
    }

    link_concepts_to_payers(concepts_kg, payers_kg, user_concepts, "compiled_kg/merged_kg.yaml")
