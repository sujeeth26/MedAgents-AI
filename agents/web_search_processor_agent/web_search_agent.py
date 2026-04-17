import requests
from typing import Dict

from .tavily_search import TavilySearchAgent, DuckDuckGoSearchAgent

class WebSearchAgent:
    """
    Agent responsible for retrieving real-time medical information from web sources.
    """

    def __init__(self, config):
        # Try Tavily first, fall back to DuckDuckGo
        self.tavily_search_agent = TavilySearchAgent()
        self.duckduckgo_search_agent = DuckDuckGoSearchAgent()

    def search(self, query: str) -> str:
        """
        Perform web searches using available search engines.
        """
        # print(f"[WebSearchAgent] Searching for: {query}")

        # Try Tavily first
        try:
            tavily_results = self.tavily_search_agent.search_tavily(query=query)
            if tavily_results and not tavily_results.startswith("Error"):
                return f"Web Search Results:\n{tavily_results}"
        except Exception as e:
            print(f"Tavily search failed: {e}")

        # Fall back to DuckDuckGo
        try:
            duckduckgo_results = self.duckduckgo_search_agent.search_duckduckgo(query=query)
            return f"Web Search Results:\n{duckduckgo_results}"
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            return f"Web Search Results:\nError retrieving search results: {e}"
