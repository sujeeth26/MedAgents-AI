import os
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class WebSearchProcessor:
    """
    Processes web search results and routes them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Build the prompt for the web search.
        
        Args:
            query: User query
            chat_history: chat history
            
        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        # print("Chat History:", chat_history)
            
        # Build the prompt
        prompt = f"""Here are the last few messages from our conversation:

        {chat_history}

        The user asked the following question:

        {query}

        Summarize them into a single, well-formed question only if the past conversation seems relevant to the current query so that it can be used for a web search.
        Keep it concise and ensure it captures the key intent behind the discussion.
        """

        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Fetches web search results, processes them using LLM, and returns a user-friendly response.
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        # print("Web Search Query Prompt:", web_search_query_prompt)
        web_search_query = self.llm.invoke(web_search_query_prompt)
        # print("Web Search Query:", web_search_query)
        
        # Retrieve web search results
        web_results = self.web_search_agent.search(web_search_query.content)

        # print(f"[WebSearchProcessor] Fetched results: {web_results}")
        
        # Construct prompt to LLM for processing the results with proper formatting and citations
        llm_prompt = f"""You are a compassionate and professional medical doctor providing clear, accurate medical information to patients.

🌟 **RESPONSE REQUIREMENTS:**
- **Concise & Clear** - Keep responses focused and easy to read
- **Polite & Empathetic** - Speak with warmth and genuine care for the patient's wellbeing
- **Clear & Simple** - Use everyday language, avoid medical jargon or explain it simply
- **Informative** - Provide essential facts without overwhelming details
- **Professional** - Speak as a caring physician would - confident yet humble

📋 **CONTENT GUIDELINES:**
- Answer the query directly and helpfully
- Include key facts from reliable sources with proper citations
- Cite sources using [1], [2], etc. format in the text
- **MANDATORY:** At the end of your response, list the sources with their Titles and URLs.
- Format: 
  **Sources:**
  1. [Title](URL)
  2. [Title](URL)
- If uncertain, acknowledge limitations and suggest professional consultation
- Always prioritize patient safety and well-being

🔍 **User Query:** {query}

📚 **Medical Information from Reliable Sources:**
{web_results}

📝 **Your Response (as a caring physician, include citations):**"""

        # Invoke the LLM to process the results with error handling
        try:
            response = self.llm.invoke(llm_prompt)

            # Ensure response is not too long (increased limit)
            response_text = response.content if hasattr(response, 'content') else str(response)
            # if len(response_text) > 1000:
            #     response_text = response_text[:997] + "..."

            return response_text

        except Exception as e:
            print(f"[WebSearchProcessor] LLM error: {e}")
            import traceback
            traceback.print_exc()
            
            # Check if it's a rate limit error
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                return """I apologize, but I'm currently experiencing high API usage and cannot process web searches at the moment. 

Please try one of these alternatives:
1. Wait a few minutes and try again
2. Search directly on medical websites like Mayo Clinic, WebMD, or CDC
3. Ask me to answer from my existing medical knowledge (RAG agent)

Your question: "{query}" - I can try to answer from my medical knowledge base if you'd like.""".format(query=query)
            
            # Fallback to a simple summary if LLM fails
            if web_results and len(web_results) > 0:
                # Extract key information from search results
                fallback_response = f"""Based on current web search results, here's what I found about "{query}":

🔍 **Key Findings:**
{web_results[:300]}

💡 **Recommendation:** For the most current and detailed information, I recommend consulting reliable medical sources or speaking with a healthcare professional. The information above is based on recent web searches and may evolve as new research emerges."""

                return fallback_response[:300] + "..." if len(fallback_response) > 300 else fallback_response
            else:
                return f"Unable to retrieve current information about '{query}'. Please consult with a healthcare professional for the latest medical guidance."
