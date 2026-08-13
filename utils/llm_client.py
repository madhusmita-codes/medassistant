"""
Groq LLM client — free API, extremely fast inference.
Sign up at https://console.groq.com to get your free API key.
Uses llama-3.3-70b-versatile model (free tier).
"""

from langchain_groq import ChatGroq
import os


def get_llm(temperature: float = 0.3) -> ChatGroq:
    """Return a configured Groq LLM client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Get a free key at https://console.groq.com and add it to your .env file."
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",  # Best free model on Groq
        temperature=temperature,
        groq_api_key=api_key,
    )
