from collections import defaultdict

def group_clauses_by_evidence_category(clauses):

    grouped = defaultdict(list)

    for clause in clauses:
        category = getattr(clause, "evidence_category", "uncategorized")
        grouped[category].append(clause)

    results = []

    for category, items in grouped.items():

        satisfied = any(c.status for c in items)

        results.append({
            "evidence_category": category,
            "status": "Sufficient" if satisfied else "Insufficient",
            "clauses": items
        })

    return results