"""
Shared state for the MedAssist LangGraph agent pipeline.
Every agent reads from and writes to this state object.
"""

from typing import TypedDict, List, Optional


class MedAssistState(TypedDict):
    # --- Input ---
    user_input: str                        # Raw symptom description from user

    # --- Triage Agent output ---
    extracted_symptoms: List[str]          # ["fever", "headache", "neck stiffness"]
    triage_severity: str                   # "low" | "medium" | "high" | "emergency"
    search_query: str                      # Optimised query for PubMed
    triage_confidence: float               # 0.0 - 1.0
    needs_clarification: bool             # If True, loop back and ask user

    # --- Research Agent output ---
    articles: List[dict]                   # List of PubMed articles
    abstracts: List[str]                   # Full abstract texts

    # --- Reasoning Agent output ---
    analysis: str                          # LLM cross-reference analysis
    possible_conditions: List[str]         # ["Meningitis", "Migraine", ...]
    red_flags: List[str]                   # Warning signs found

    # --- Report Agent output ---
    final_report: str                      # Full formatted clinical summary

    # --- Control ---
    error: Optional[str]                   # Any error message
    clarification_question: Optional[str]  # Question to ask user if unclear
