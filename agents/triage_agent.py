"""
Triage Agent
------------
First agent in the pipeline. Takes raw user text and:
1. Extracts structured symptoms
2. Assesses severity (low / medium / high / emergency)
3. Builds an optimised PubMed search query
4. Flags if input is too vague and needs clarification
"""

import json
from utils.state import MedAssistState
from utils.llm_client import get_llm


TRIAGE_PROMPT = """You are a medical triage AI assistant. Analyse the patient's description and respond ONLY with a valid JSON object — no markdown, no explanation, just the JSON.

Patient description: {user_input}

Respond with this exact JSON structure:
{{
  "extracted_symptoms": ["symptom1", "symptom2"],
  "triage_severity": "low|medium|high|emergency",
  "search_query": "optimised pubmed search terms",
  "triage_confidence": 0.85,
  "needs_clarification": false,
  "clarification_question": null
}}

Severity guide:
- emergency: chest pain, stroke signs, severe breathing difficulty, unconsciousness
- high: high fever with stiff neck, severe abdominal pain, signs of infection
- medium: persistent fever, moderate pain, concerning but stable symptoms  
- low: mild cold, minor aches, general wellness questions

If the description is too vague to make any assessment, set needs_clarification to true and provide a specific clarification_question."""


def triage_agent(state: MedAssistState) -> MedAssistState:
    """Extract symptoms and assess severity from user input."""
    print("🔍 [Triage Agent] Analysing symptoms...")

    llm = get_llm(temperature=0.1)  # Low temp for structured extraction
    prompt = TRIAGE_PROMPT.format(user_input=state["user_input"])

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Clean up any accidental markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)

        return {
            **state,
            "extracted_symptoms": data.get("extracted_symptoms", []),
            "triage_severity": data.get("triage_severity", "medium"),
            "search_query": data.get("search_query", state["user_input"]),
            "triage_confidence": float(data.get("triage_confidence", 0.5)),
            "needs_clarification": bool(data.get("needs_clarification", False)),
            "clarification_question": data.get("clarification_question"),
            "error": None,
        }

    except json.JSONDecodeError as e:
        # If JSON parsing fails, build a basic fallback state
        return {
            **state,
            "extracted_symptoms": [],
            "triage_severity": "medium",
            "search_query": state["user_input"],
            "triage_confidence": 0.3,
            "needs_clarification": False,
            "error": f"Triage parsing error: {e}",
        }
    except Exception as e:
        return {**state, "error": f"Triage agent error: {e}"}
