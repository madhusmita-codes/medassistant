"""
Research Agent
--------------
Second agent in the pipeline.
Uses the optimised search query from Triage Agent to:
1. Fetch relevant PubMed articles (free API)
2. Retrieve full abstracts for top articles
3. Store results in state for the Reasoning Agent
"""

from utils.state import MedAssistState
from tools.pubmed_tool import search_pubmed, fetch_abstract


def research_agent(state: MedAssistState) -> MedAssistState:
    """Fetch relevant medical research from PubMed."""
    print("📚 [Research Agent] Searching PubMed for relevant studies...")

    query = state.get("search_query", state["user_input"])
    symptoms = state.get("extracted_symptoms", [])

    # Build a richer query using extracted symptoms if available
    if symptoms:
        symptom_str = " ".join(symptoms[:4])  # Use top 4 symptoms
        full_query = f"{symptom_str} diagnosis treatment"
    else:
        full_query = query

    print(f"   Query: '{full_query}'")

    # Fetch articles from PubMed
    articles = search_pubmed(full_query, max_results=5)

    if not articles or "error" in articles[0]:
        return {
            **state,
            "articles": [],
            "abstracts": [],
            "error": "Could not fetch PubMed articles. Check your internet connection.",
        }

    print(f"   Found {len(articles)} articles")

    # Fetch abstracts for top 3 articles
    abstracts = []
    for article in articles[:3]:
        pmid = article.get("pmid", "")
        if pmid:
            print(f"   Fetching abstract for PMID {pmid}...")
            abstract = fetch_abstract(pmid)
            abstracts.append(f"[{article['title']}]\n{abstract}")

    return {
        **state,
        "articles": articles,
        "abstracts": abstracts,
        "error": None,
    }
