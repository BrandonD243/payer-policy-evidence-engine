import json
from typing import List, Dict
from openai import OpenAI

client = OpenAI()


def extract_concepts_from_text(
    document_text: str,
    relevant_concepts: List[Dict],
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Returns:
    [
        {
            "concept_id": "spinal_stenosis",
            "evidence_text": "Lumbar spinal stenosis with left-sided radiculopathy",
            "confidence": "strong",
            "certainty_level": "confirmed"
        }
    ]
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
        parsed = json.loads(raw_output)
        mentions = parsed.get("concept_mentions", [])

        # Rename text_span → evidence_text
        for m in mentions:
            m["evidence_text"] = m.pop("text_span")

        return mentions

    except json.JSONDecodeError:
        print("⚠️ OpenAI returned invalid JSON:")
        print(raw_output)
        return []