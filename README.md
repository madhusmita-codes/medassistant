# 🏥 MedAssist AI
### Autonomous Medical Research & Symptom Analysis Agent

Built with **LangGraph** · **Groq (Llama 3, free)** · **PubMed API (free)** · **AgentOps** · **Streamlit**

---

## What It Does

MedAssist AI is a multi-agent system that:
1. **Triages** your symptoms and assesses severity
2. **Searches PubMed** (real medical database) for relevant clinical research
3. **Reasons** over the research to identify possible conditions and red flags
4. **Generates** a structured clinical summary report

All powered by **free APIs** — no credit card, no subscription needed.

---

## Agent Architecture (LangGraph)

```
User Input
    ↓
[Triage Agent]  ──── unclear input? ──→  Ask for clarification → END
    ↓
[Research Agent]  ←── PubMed API (free)
    ↓
[Reasoning Agent]  ←── Groq LLM (free, Llama 3)
    ↓
[Report Agent]
    ↓
Final Report + Sources
```

AgentOps tracks every LLM call, agent hop, latency, and token usage.

---

## Setup (5 minutes)

### 1. Clone / download this project

```bash
cd medassist_ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get your free API keys

| Service | Where to get it | Cost |
|---|---|---|
| **Groq** | https://console.groq.com → Sign up → API Keys | Free |
| **AgentOps** | https://app.agentops.ai → Sign up | Free |
| **PubMed** | No key needed! | Always free |

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:
```
GROQ_API_KEY=gsk_your_key_here
AGENTOPS_API_KEY=your_key_here
PUBMED_EMAIL=your_email@gmail.com
```

### 5. Run the app

```bash
# Web UI (recommended)
streamlit run app.py

# OR command line
python pipeline.py
```

The app opens at **http://localhost:8501**

---

## Project Structure

```
medassist_ai/
│
├── app.py                  # Streamlit web interface
├── pipeline.py             # LangGraph graph — main orchestrator
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── triage_agent.py     # Symptom extraction + severity assessment
│   ├── research_agent.py   # PubMed API search + abstract fetching
│   ├── reasoning_agent.py  # LLM cross-reference analysis
│   └── report_agent.py     # Final structured report generation
│
├── tools/
│   └── pubmed_tool.py      # PubMed API wrapper
│
└── utils/
    ├── state.py            # Shared LangGraph state (TypedDict)
    └── llm_client.py       # Groq LLM client setup
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Agent orchestration | LangGraph StateGraph | Conditional routing, real agentic flow |
| LLM | Groq + Llama 3.3 70B | Fast, free, high quality |
| Medical data | PubMed API | Real clinical research, completely free |
| Observability | AgentOps | Tracks every LLM call and agent hop |
| Frontend | Streamlit | Simple, runs locally, no deployment needed |

---

## Example Queries to Try

- `"High fever 103°F for 3 days, severe headache, stiff neck, sensitivity to light"`
- `"Persistent chest pain radiating to left arm, shortness of breath, sweating"`
- `"Lower right abdominal pain that worsens with movement, nausea, low-grade fever"`
- `"Dry cough for 2 weeks, mild fever, fatigue, loss of smell and taste"`

---

## GitHub Setup

```bash
git init
git add .
git commit -m "Initial commit: MedAssist AI multi-agent system"
git remote add origin https://github.com/YOUR_USERNAME/medassist-ai.git
git push -u origin main
```

---

> ⚕️ **Medical Disclaimer:** This tool is for educational and research purposes only. It does not provide medical diagnoses and is not a substitute for professional medical advice. Always consult a qualified healthcare provider for medical concerns.
