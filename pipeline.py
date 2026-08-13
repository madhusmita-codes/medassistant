"""
MedAssist AI — Main LangGraph Pipeline
=======================================
Orchestrates 4 agents using LangGraph StateGraph with conditional routing.

Flow:
  START
    ↓
  triage_agent  ──── needs_clarification=True ──→  END (ask user)
    ↓ (confidence OK)
  research_agent
    ↓
  reasoning_agent
    ↓
  report_agent
    ↓
  END
"""

import os
import agentops
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from utils.state import MedAssistState
from agents.triage_agent import triage_agent
from agents.research_agent import research_agent
from agents.reasoning_agent import reasoning_agent
from agents.report_agent import report_agent

load_dotenv()


def should_clarify(state: MedAssistState) -> str:
    """
    Conditional edge after triage.
    If the agent needs more info from the user, stop and ask.
    Otherwise continue to research.
    """
    if state.get("needs_clarification") or state.get("error"):
        return "end"
    return "research"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent pipeline."""
    graph = StateGraph(MedAssistState)

    # Register all agent nodes
    graph.add_node("triage", triage_agent)
    graph.add_node("research", research_agent)
    graph.add_node("reasoning", reasoning_agent)
    graph.add_node("report", report_agent)

    # Entry point
    graph.set_entry_point("triage")

    # Conditional routing after triage
    graph.add_conditional_edges(
        "triage",
        should_clarify,
        {
            "end": END,
            "research": "research",
        },
    )

    # Linear flow after research
    graph.add_edge("research", "reasoning")
    graph.add_edge("reasoning", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_medassist(user_input: str) -> MedAssistState:
    """
    Run the full MedAssist pipeline for a given user input.
    AgentOps automatically tracks all LLM calls and agent transitions.
    """
    # Initialise AgentOps session (tracks everything automatically)
    agentops_key = os.getenv("AGENTOPS_API_KEY")
    if agentops_key:
        agentops.init(agentops_key, tags=["medassist", "medical-ai", "langgraph"])
        print("✅ AgentOps session started — tracking all LLM calls")
    else:
        print("⚠️  AGENTOPS_API_KEY not set — running without observability tracking")

    # Build the graph
    app = build_graph()

    # Initial state
    initial_state: MedAssistState = {
        "user_input": user_input,
        "extracted_symptoms": [],
        "triage_severity": "",
        "search_query": "",
        "triage_confidence": 0.0,
        "needs_clarification": False,
        "articles": [],
        "abstracts": [],
        "analysis": "",
        "possible_conditions": [],
        "red_flags": [],
        "final_report": "",
        "error": None,
        "clarification_question": None,
    }

    print(f"\n{'='*60}")
    print("🏥 MedAssist AI — Starting analysis...")
    print(f"{'='*60}\n")

    # Run the graph
    final_state = app.invoke(initial_state)

    # End AgentOps session
    if agentops_key:
        agentops.end_session("Success")

    return final_state


if __name__ == "__main__":
    # Quick CLI test
    test_input = "I have had a high fever of 103°F for 3 days, severe headache, stiff neck, and sensitivity to light."
    result = run_medassist(test_input)

    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(result.get("final_report", "No report generated"))

    if result.get("needs_clarification"):
        print("\n❓ CLARIFICATION NEEDED:")
        print(result.get("clarification_question"))
