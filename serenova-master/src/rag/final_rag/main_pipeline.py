from src.rag.final_rag.retriever import get_relevant_rules
from src.rag.final_rag.llm_engine import generate_explanation


def run_rag_pipeline(input_data):

    conditions = input_data.get("conditions", [])
    risk_score = input_data.get("risk_score", 0)
    week = input_data.get("gestational_week", "NA")

    important = [c for c in conditions if c["status"] in ["high", "watch"]]

    summary = f"""
Week: {week}
Risk score: {risk_score}
Important conditions: {important}
"""

    rules_list = get_relevant_rules(summary)

    if not rules_list:
        return {
            "summary": summary.strip(),
            "clinical_findings": [],
            "explanation": "No rules found"
        }

    seen = set()
    clinical_findings = []

    for doc in rules_list:
        line = doc.page_content.split("\n")[0].strip()

        # ❌ remove useless generic rules
        if "Multi-Parameter Risk" in line:
            continue

        if line not in seen:
            seen.add(line)
            clinical_findings.append(line)

    rules_text = "\n".join([doc.page_content for doc in rules_list])

    raw_explanation = generate_explanation(summary, rules_text)

    explanation = raw_explanation.split("\n")[0].strip()

    return {
        "summary": summary.strip(),
        "clinical_findings": clinical_findings,
        "explanation": explanation
    }