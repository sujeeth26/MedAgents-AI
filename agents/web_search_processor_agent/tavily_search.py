import requests
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Any

class DuckDuckGoSearchAgent:
    """
    Web search agent using DuckDuckGo search engine.
    """

    def __init__(self):
        """Initialize the DuckDuckGo search agent."""
        pass

    def search_duckduckgo(self, query: str) -> str:
        """Perform a web search using DuckDuckGo."""

        # DuckDuckGo search URL
        search_url = "https://html.duckduckgo.com/html/"

        # Headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Parameters for the search
        params = {
            'q': query,
            'kl': 'us-en',
            'kc': '1'  # Safe search off for medical content
        }

        try:
            # Make the request
            response = requests.get(search_url, params=params, headers=headers, timeout=10)

            if response.status_code != 200:
                return f"Error: HTTP {response.status_code} when searching"

            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all result links and titles
            results = []
            result_divs = soup.find_all('div', class_='result')

            for i, result in enumerate(result_divs[:5], 1):  # Get top 5 results
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')

                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem['href']
                    snippet = snippet_elem.get_text(strip=True)

                    # Clean up the URL (remove DuckDuckGo tracking)
                    if url.startswith('//'):
                        url = 'https:' + url
                    elif url.startswith('/'):
                        url = 'https://duckduckgo.com' + url

                    results.append(f"[{i}] {title}\nURL: {url}\nContent: {snippet[:200]}...\nRelevance: High")

            if results:
                return "\n\n".join(results)
            else:
                return "No relevant results found."

        except requests.exceptions.RequestException as e:
            return f"Error retrieving web search results: {e}"
        except Exception as e:
            return f"Error processing search results: {e}"

class TavilySearchAgent:
    """
    Fallback to Tavily search if available.
    """
    def __init__(self):
        """
        Initialize the Tavily search agent.
        """
        pass

    def search_tavily(self, query: str) -> str:
        """Perform a general web search using Tavily API."""
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tavily_search = TavilySearchResults(max_results=5)

            # Strip any surrounding quotes from the query
            query = query.strip('"\'')
            search_docs = tavily_search.invoke(query)

            if len(search_docs):
                results = []
                for i, res in enumerate(search_docs, 1):
                    results.append(f"[{i}] {res['title']}\nURL: {res['url']}\nContent: {res['content'][:200]}...\nRelevance Score: {res['score']:.2f}")
                return "\n\n".join(results)
            return "No relevant results found."
        except Exception as e:
            # Fall back to DuckDuckGo if Tavily fails
            duckduckgo = DuckDuckGoSearchAgent()
            return duckduckgo.search_duckduckgo(query)