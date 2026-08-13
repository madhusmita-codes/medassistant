"""
MedAssist AI — Streamlit Web Interface
Run with: streamlit run app.py
"""

import streamlit as st
from pipeline import run_medassist

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MedAssist AI",
    page_icon="🏥",
    layout="centered",
)

# ── Header ───────────────────────────────────────────────────
st.title("🏥 MedAssist AI")
st.caption("Autonomous Medical Research & Symptom Analysis Agent")
st.markdown(
    "> ⚕️ **Disclaimer:** This tool provides AI-assisted research summaries only. "
    "It is **not** a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider."
)
st.divider()

# ── Example symptoms ─────────────────────────────────────────
with st.expander("💡 Try an example"):
    examples = [
        "I have had a high fever of 103°F for 3 days, severe headache, stiff neck, and sensitivity to light.",
        "I've been experiencing persistent chest pain radiating to my left arm, shortness of breath, and sweating.",
        "I have abdominal pain on the lower right side that gets worse when I move, along with nausea and low-grade fever.",
        "I've had a dry cough for 2 weeks, mild fever, fatigue, and loss of smell and taste.",
    ]
    for ex in examples:
        if st.button(ex[:80] + "...", key=ex):
            st.session_state["symptom_input"] = ex

# ── Input ─────────────────────────────────────────────────────
symptom_input = st.text_area(
    "Describe your symptoms in detail:",
    value=st.session_state.get("symptom_input", ""),
    height=120,
    placeholder="e.g. I have had a fever for 3 days, severe headache, and stiff neck...",
)

analyse_btn = st.button("🔍 Analyse Symptoms", type="primary", use_container_width=True)

# ── Run pipeline ─────────────────────────────────────────────
if analyse_btn:
    if not symptom_input.strip():
        st.warning("Please describe your symptoms first.")
    else:
        with st.spinner("Running MedAssist AI agents..."):
            # Progress indicators
            progress = st.empty()
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.info("🔍 Triage")
            with col2:
                st.info("📚 Research")
            with col3:
                st.info("🧠 Reasoning")
            with col4:
                st.info("📝 Report")

            try:
                result = run_medassist(symptom_input)
                progress.empty()

                # ── Clarification needed ──────────────────────────
                if result.get("needs_clarification"):
                    st.warning("❓ I need more information to help you accurately:")
                    st.info(result.get("clarification_question", "Please provide more details."))

                # ── Error ────────────────────────────────────────
                elif result.get("error"):
                    st.error(f"An error occurred: {result['error']}")

                # ── Success ───────────────────────────────────────
                else:
                    # Severity badge
                    severity = result.get("triage_severity", "unknown")
                    severity_colors = {
                        "emergency": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🟢",
                    }
                    badge = severity_colors.get(severity, "⚪")
                    st.markdown(f"### {badge} Severity: **{severity.upper()}**")

                    # Symptoms extracted
                    symptoms = result.get("extracted_symptoms", [])
                    if symptoms:
                        st.markdown("**Symptoms identified:** " + " · ".join(f"`{s}`" for s in symptoms))

                    st.divider()

                    # Possible conditions
                    conditions = result.get("possible_conditions", [])
                    if conditions:
                        st.markdown("### 🩺 Possible Conditions")
                        for c in conditions:
                            st.markdown(f"- {c}")

                    # Red flags
                    red_flags = result.get("red_flags", [])
                    if red_flags:
                        st.markdown("### ⚠️ Red Flag Warnings")
                        for r in red_flags:
                            st.error(f"⚠️ {r}")

                    st.divider()

                    # Full report
                    st.markdown("### 📋 Full Clinical Summary")
                    st.markdown(result.get("final_report", "No report generated."))

                    st.divider()

                    # Research sources
                    articles = result.get("articles", [])
                    if articles:
                        st.markdown("### 📚 Research Sources (PubMed)")
                        for a in articles:
                            title = a.get("title", "Unknown")
                            url = a.get("url", "")
                            journal = a.get("journal", "")
                            date = a.get("date", "")
                            st.markdown(f"- [{title}]({url}) — *{journal}, {date}*")

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.info("Make sure your GROQ_API_KEY is set in your .env file.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with LangGraph · Groq (Llama 3) · PubMed API · AgentOps · Streamlit | "
    "By Madhusmita Singh"
)
