import json
import re
import string
from typing import List, Dict
from openai import OpenAI

client = OpenAI()


def extract_concepts_from_text(
    document_text: str,
    relevant_concepts: List[Dict],
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Extract concept mentions using LLM, returning numeric confidence and categorical certainty_level.
    Confidence is a float between 0.0 and 1.0 based on evidence strength.
    """

    if not document_text:
        return []

    concepts_json = json.dumps(relevant_concepts, indent=2)

    prompt = f"""
You are a clinical text analysis engine.

TASK:
Identify mentions of the following medical concepts in the document.

RULES:
- Only extract text that explicitly supports a concept.
- Ignore negated mentions.
- Use exact text spans from the document.
- Assign:
  - confidence as a decimal float between 0.01 and 0.99 based on strength of evidence:
  - certainty_level: confirmed | suspected | planned
- Output valid JSON only.
- Do NOT hallucinate concepts. Only extract concepts with supporting evidence.

OUTPUT FORMAT:
{{
  "concept_mentions": [
    {{
      "concept_id": "<concept_id>",
      "text_span": "<exact quote>",
      "confidence": <decimal float between 0.01 and 0.99>,
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
            concept_id = m.get("concept_id")
            text_span = m.get("text_span")
            confidence = m.get("confidence")

            if not concept_id or not text_span:
                continue

            # STRICT: enforce numeric confidence
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                print(f"⚠️ Non-numeric confidence detected for concept {concept_id}, skipping.")
                continue

            # Build short-id lookup from registry
            registry_lookup = {
                c["concept_id"].split(".")[-1]: c
                for c in relevant_concepts
            }

            concept_def = registry_lookup.get(concept_id)

            if concept_def is None:
                print(f"Unknown concept returned by extractor: {concept_id}")
                continue


            print("\n--- VALIDATION DEBUG ---")
            print("concept_id:", concept_id)
            print("text_span:", text_span)
            print("indicators:",
                concept_def.get("positive_indicators", []) +
                concept_def.get("therapy_indicators", []) +
                concept_def.get("duration_indicators", [])
            )

            if indicator_match(text_span, concept_def):

                validated_mentions.append({
                    "concept_id": concept_id,
                    "evidence_text": text_span,
                    "confidence": min(max(confidence, 0.0), 1.0),  # clamp to [0.0, 1.0]
                    "certainty_level": m.get("certainty_level", "confirmed")
                })
            else:
                print(f"Filtered hallucinated concept: {concept_id}")

        return validated_mentions

    except json.JSONDecodeError:
        print("⚠️ OpenAI returned invalid JSON:")
        print(raw_output)
        return []


def safe_json_load(text: str):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    return json.loads(text)


def normalize(text: str) -> str:
    return (
        text.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .strip()
    )


def singularize(token: str) -> str:
    # very lightweight clinical normalization
    if token.endswith("es"):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def indicator_match(evidence_text: str, concept_def: Dict) -> bool:
    if not evidence_text:
        return False

    evidence_text_norm = normalize(evidence_text)
    evidence_tokens = {
        singularize(t) for t in evidence_text_norm.split()
    }

    indicators = []
    indicators.extend(concept_def.get("positive_indicators", []))
    indicators.extend(concept_def.get("therapy_indicators", []))
    indicators.extend(concept_def.get("duration_indicators", []))

    # Derived concepts (no indicators) are allowed
    if not indicators:
        return True

    for ind in indicators:
        ind_norm = normalize(ind)
        ind_tokens = {singularize(t) for t in ind_norm.split()}

        # 1. Exact substring match
        if ind_norm in evidence_text_norm:
            return True

        # 2. Strong keyword match (ANY meaningful token)
        if any(token in evidence_tokens for token in ind_tokens):
            return True

        # 3. Token coverage match (>=50%)
        overlap = len(ind_tokens & evidence_tokens) / max(len(ind_tokens), 1)
        if overlap >= 0.5:
            return True

    return False