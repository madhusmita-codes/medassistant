"""
Reasoning Agent
---------------
Third agent in the pipeline.
Cross-references the patient's symptoms against retrieved PubMed abstracts using LLM.
Identifies possible conditions and red flag warning signs.
"""

from utils.state import MedAssistState
from utils.llm_client import get_llm


REASONING_PROMPT = """You are a clinical reasoning AI assistant helping analyse a patient's symptoms against recent medical research. This is for informational purposes only and does not replace professional medical advice.

PATIENT SYMPTOMS:
{symptoms}

SEVERITY ASSESSMENT: {severity}

RELEVANT MEDICAL RESEARCH:
{abstracts}

Based on the symptoms and the research above, provide:
1. A list of 2-4 possible conditions that match these symptoms (most likely first)
2. Any red flag warning signs present that need urgent attention
3. A brief clinical reasoning summary (3-5 sentences) explaining the connection between the symptoms and the research findings

Format your response clearly with these exact headers:
POSSIBLE CONDITIONS:
RED FLAGS:
CLINICAL REASONING:"""


def reasoning_agent(state: MedAssistState) -> MedAssistState:
    """Cross-reference symptoms against research to identify conditions."""
    print("🧠 [Reasoning Agent] Analysing symptoms against research...")

    symptoms = state.get("extracted_symptoms", [])
    severity = state.get("triage_severity", "unknown")
    abstracts = state.get("abstracts", [])

    if not abstracts:
        return {
            **state,
            "analysis": "No research data available to reason against.",
            "possible_conditions": ["Unable to determine — no research fetched"],
            "red_flags": [],
        }

    symptom_str = ", ".join(symptoms) if symptoms else state["user_input"]
    abstract_str = "\n\n---\n\n".join(abstracts[:3])  # Use top 3 abstracts

    prompt = REASONING_PROMPT.format(
        symptoms=symptom_str,
        severity=severity,
        abstracts=abstract_str[:4000],  # Cap to avoid token limits
    )

    llm = get_llm(temperature=0.4)
    response = llm.invoke(prompt)
    analysis_text = response.content.strip()

    # Parse out conditions and red flags
    possible_conditions = []
    red_flags = []

    lines = analysis_text.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        if "POSSIBLE CONDITIONS:" in line:
            current_section = "conditions"
        elif "RED FLAGS:" in line:
            current_section = "red_flags"
        elif "CLINICAL REASONING:" in line:
            current_section = "reasoning"
        elif line.startswith("-") or line.startswith("•") or (line and line[0].isdigit()):
            clean = line.lstrip("-•0123456789. ").strip()
            if clean:
                if current_section == "conditions":
                    possible_conditions.append(clean)
                elif current_section == "red_flags":
                    red_flags.append(clean)

    return {
        **state,
        "analysis": analysis_text,
        "possible_conditions": possible_conditions or ["See full analysis below"],
        "red_flags": red_flags,
        "error": None,
    }
