"""
PubMed API Tool — completely free, no subscription needed.
Fetches real clinical research articles based on symptom/disease queries.
"""

import requests
import os
from typing import List, Dict


PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMAIL = os.getenv("PUBMED_EMAIL", "medassist@research.ai")


def search_pubmed(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search PubMed for medical articles related to the query.
    Returns list of articles with title, abstract, and URL.
    """
    try:
        # Step 1: Search for article IDs
        search_url = f"{PUBMED_BASE}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "tool": "MedAssistAI",
            "email": EMAIL,
        }
        search_resp = requests.get(search_url, params=search_params, timeout=10)
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return []

        # Step 2: Fetch article summaries
        fetch_url = f"{PUBMED_BASE}/esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
            "tool": "MedAssistAI",
            "email": EMAIL,
        }
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=10)
        fetch_resp.raise_for_status()
        result = fetch_resp.json().get("result", {})

        articles = []
        for pmid in ids:
            if pmid not in result:
                continue
            article = result[pmid]
            title = article.get("title", "No title")
            authors = article.get("authors", [])
            author_str = authors[0].get("name", "Unknown") + " et al." if authors else "Unknown"
            pub_date = article.get("pubdate", "Unknown date")
            source = article.get("source", "Unknown journal")
            articles.append({
                "pmid": pmid,
                "title": title,
                "authors": author_str,
                "journal": source,
                "date": pub_date,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })

        return articles

    except Exception as e:
        return [{"error": str(e), "title": "PubMed fetch failed", "url": ""}]


def fetch_abstract(pmid: str) -> str:
    """Fetch the full abstract text for a given PubMed article ID."""
    try:
        url = f"{PUBMED_BASE}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "text",
            "tool": "MedAssistAI",
            "email": EMAIL,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        # Return first 1500 chars to stay within LLM context limits
        return resp.text.strip()[:1500]
    except Exception as e:
        return f"Could not fetch abstract: {e}"
