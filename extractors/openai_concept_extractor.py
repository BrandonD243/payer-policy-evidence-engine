import json
import re
from typing import List, Dict
from openai import OpenAI

client = OpenAI()

def extract_concepts_from_text(
    document_text: str,
    relevant_concepts: List[Dict],
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Extracts concept mentions from a document text.
    Returns a list of dicts with keys:
    - concept_id
    - evidence_text
    - confidence
    - certainty_level
    """

    concepts_json = json.dumps(relevant_concepts, indent=2)

    prompt = f"""
You are a clinical text analysis engine.

TASK:
Identify mentions of the following medical concepts in the document.

RULES:
- Only extract text that explicitly supports a concept
- Ignore negated or contradicted mentions
- Use exact text spans from the document
- Assign:
  - confidence: weak | moderate | strong
  - certainty_level: confirmed | suspected | planned
- Do NOT hallucinate concepts
- Output valid JSON only

OUTPUT FORMAT:
{{
  "concept_mentions": [
    {{
      "concept_id": "<concept_id>",
      "text_span": "<exact quote>",
      "confidence": "weak | moderate | strong",
      "certainty_level": "confirmed | suspected | planned"
    }}
  ]
}}

DOCUMENT:
\"\"\"{document_text}\"\"\"

CONCEPT REGISTRY:
{concepts_json}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You extract structured clinical evidence."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        parsed = safe_json_load(raw_output)

        if not parsed or "concept_mentions" not in parsed:
            print("⚠️ No valid concept_mentions returned")
            return []

        mentions = parsed["concept_mentions"]


        validated_mentions = []

        for m in mentions:
            m["evidence_text"] = m.pop("text_span")

            concept_id = m.get("concept_id")
            concept_def = next(
                (c for c in relevant_concepts if c["id"] == concept_id),
                None
            )

            if concept_def and indicator_match(m["evidence_text"], concept_def):
                validated_mentions.append(m)
            else:
                print(f"Filtered hallucinated concept: {concept_id}")

        return validated_mentions


    except json.JSONDecodeError:
        print("⚠️ OpenAI returned invalid JSON:")
        print(raw_output)
        return []

def safe_json_load(text: str):
    """
    Cleans OpenAI responses that include ```json blocks
    and returns parsed JSON.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

def indicator_match(evidence_text: str, concept_def: Dict) -> bool:
    """
    Ensures evidence_text contains at least one positive indicator
    defined in the concept registry.
    """
    if not evidence_text:
        return False

    evidence_text = evidence_text.lower()
    indicators = concept_def.get("positive_indicators", [])

    return any(ind.lower() in evidence_text for ind in indicators)

    return json.loads(text)


