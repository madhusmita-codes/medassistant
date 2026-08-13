"""
Report Agent
------------
Final agent in the pipeline.
Takes all outputs from previous agents and generates a clean,
structured clinical summary report for the user.
"""

from utils.state import MedAssistState
from utils.llm_client import get_llm


REPORT_PROMPT = """You are a medical report writer. Generate a clear, structured, and empathetic patient-facing medical research summary based on the following analysis.

IMPORTANT: Always remind the user this is AI-assisted research information only and NOT a medical diagnosis. Always recommend consulting a qualified doctor.

INPUT DATA:
- Symptoms identified: {symptoms}
- Severity level: {severity}
- Possible conditions: {conditions}
- Red flags: {red_flags}
- Clinical analysis: {analysis}
- Research sources: {article_titles}

Write a well-structured report with these sections:
1. 📋 SUMMARY — Brief overview of what was found
2. 🔴 SEVERITY LEVEL — Clearly state severity and what it means  
3. 🩺 POSSIBLE CONDITIONS — Explain each condition briefly in simple language
4. ⚠️ RED FLAGS — List any warning signs requiring immediate medical attention
5. 📚 RESEARCH BASIS — Mention the key studies consulted
6. ✅ RECOMMENDED NEXT STEPS — Practical advice on what the patient should do
7. ⚕️ DISCLAIMER — Standard medical disclaimer

Keep language clear, compassionate, and easy to understand for a non-medical audience."""


def report_agent(state: MedAssistState) -> MedAssistState:
    """Generate the final structured medical research report."""
    print("📝 [Report Agent] Generating final report...")

    symptoms = ", ".join(state.get("extracted_symptoms", [])) or state["user_input"]
    severity = state.get("triage_severity", "unknown")
    conditions = "\n".join(f"- {c}" for c in state.get("possible_conditions", []))
    red_flags = "\n".join(f"- {r}" for r in state.get("red_flags", [])) or "None identified"
    analysis = state.get("analysis", "No analysis available")
    articles = state.get("articles", [])
    article_titles = "\n".join(
        f"- {a.get('title', 'Unknown')} ({a.get('journal', '')} {a.get('date', '')})"
        for a in articles[:5]
    )

    prompt = REPORT_PROMPT.format(
        symptoms=symptoms,
        severity=severity,
        conditions=conditions or "Not determined",
        red_flags=red_flags,
        analysis=analysis[:2000],
        article_titles=article_titles or "No articles retrieved",
    )

    llm = get_llm(temperature=0.5)
    response = llm.invoke(prompt)

    return {
        **state,
        "final_report": response.content.strip(),
        "error": None,
    }
