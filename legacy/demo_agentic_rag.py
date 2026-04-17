"""
Demo Agentic RAG System with Pinecone and OpenRouter

This is a demonstration of an agentic RAG system that includes:
1. Query Analysis Agent - Analyzes and routes queries
2. Retrieval Agent - Retrieves relevant documents from Pinecone
3. Reflection Agent - Evaluates retrieved documents and decides if re-retrieval needed
4. Response Synthesis Agent - Generates final response

NEW: Full Multi-Agent System Demo
This demo now includes all agents:
- RAG Agent (medical knowledge)
- Web Search Agent (recent information)
- Conversation Agent (general chat)
- Chest X-ray Agent (MedRAX - 18-disease classification, segmentation, report generation, disease grounding)
- Brain Tumor Agent (MRI analysis)
- Skin Lesion Agent (classification)

Usage:
    python demo_agentic_rag.py              # Run full system demo (default)
    python demo_agentic_rag.py --rag-only    # Run RAG-only demo (backward compatibility)
"""

import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import logging

# Set up logging
logger = logging.getLogger(__name__)


class AgenticRAGSystem:
    """
    Agentic RAG system with multiple specialized agents for intelligent document retrieval
    and response generation.
    """
    
    def __init__(
        self, 
        pinecone_api_key: str,
        pinecone_index_name: str,
        openrouter_api_key: str,
        openrouter_model: str = "llama-3.3-70b-versatile"
    ):
        """
        Initialize the Agentic RAG system.
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_index_name: Name of the Pinecone index
            openrouter_api_key: LLM API key (Groq or OpenRouter)
            openrouter_model: Model to use
        """
        logger.info("🚀 Initializing Agentic RAG System...")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = pinecone_index_name
        
        # Initialize embeddings
        logger.info("📊 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        logger.info(f"🔍 Connecting to Pinecone index: {pinecone_index_name}")
        # Get the Pinecone index
        index = self.pc.Index(pinecone_index_name)
        self.vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text"
        )
        
        # Initialize LLM for different agents (using Groq)
        logger.info(f"🤖 Initializing LLM: {openrouter_model}")
        
        # Query Analysis Agent - deterministic
        self.query_analysis_llm = ChatOpenAI(
            model=openrouter_model,
            api_key=openrouter_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.1
        )
        
        # Reflection Agent - analytical
        self.reflection_llm = ChatOpenAI(
            model=openrouter_model,
            api_key=openrouter_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2
        )
        
        # Response Synthesis Agent - creative but grounded
        self.synthesis_llm = ChatOpenAI(
            model=openrouter_model,
            api_key=openrouter_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.5
        )
        
        logger.info("✅ Agentic RAG System initialized successfully!")
    
    def query_analysis_agent(self, query: str) -> Dict[str, Any]:
        """
        Agent 1: Analyzes the query and determines the search strategy.
        
        This agent:
        - Identifies query type (factual, comparison, procedural, etc.)
        - Extracts key medical terms
        - Generates search keywords
        - Suggests retrieval strategy
        
        Args:
            query: User's original query
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("🔎 Query Analysis Agent: Analyzing query...")
        
        prompt = f"""You are a query analysis expert for a medical RAG system.

Analyze the following user query and provide:
1. Query Type: (e.g., factual, comparison, procedural, diagnostic, treatment-related)
2. Key Medical Terms: Extract important medical terminology
3. Search Keywords: Generate 3-5 optimized search keywords
4. Retrieval Strategy: Suggest how many documents to retrieve (3-10)
5. Query Complexity: Rate as 'simple', 'moderate', or 'complex'

User Query: {query}

Provide your analysis in this exact format:
QUERY_TYPE: <type>
KEY_TERMS: <term1>, <term2>, <term3>
SEARCH_KEYWORDS: <keyword1>, <keyword2>, <keyword3>
RETRIEVAL_COUNT: <number>
COMPLEXITY: <level>
REASONING: <brief explanation>
"""
        
        response = self.query_analysis_llm.invoke([HumanMessage(content=prompt)])
        analysis_text = response.content
        
        # Parse the response
        analysis = self._parse_query_analysis(analysis_text)
        analysis['original_query'] = query
        
        logger.info(f"   ✓ Query Type: {analysis.get('query_type', 'unknown')}")
        logger.info(f"   ✓ Complexity: {analysis.get('complexity', 'unknown')}")
        logger.info(f"   ✓ Retrieval Count: {analysis.get('retrieval_count', 5)}")
        
        return analysis
    
    def _parse_query_analysis(self, text: str) -> Dict[str, Any]:
        """Parse the query analysis response."""
        analysis = {}
        
        lines = text.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace('_', '_')
                value = value.strip()
                
                if key == 'query_type' or key == 'query type':
                    analysis['query_type'] = value
                elif key == 'key_terms' or key == 'key terms':
                    analysis['key_terms'] = [term.strip() for term in value.split(',')]
                elif key == 'search_keywords' or key == 'search keywords':
                    analysis['search_keywords'] = [kw.strip() for kw in value.split(',')]
                elif key == 'retrieval_count' or key == 'retrieval count':
                    try:
                        analysis['retrieval_count'] = int(value)
                    except:
                        analysis['retrieval_count'] = 5
                elif key == 'complexity':
                    analysis['complexity'] = value
                elif key == 'reasoning':
                    analysis['reasoning'] = value
        
        # Set defaults
        if 'retrieval_count' not in analysis:
            analysis['retrieval_count'] = 5
        if 'complexity' not in analysis:
            analysis['complexity'] = 'moderate'
            
        return analysis
    
    def retrieval_agent(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Agent 2: Retrieves relevant documents from Pinecone based on analysis.
        
        This agent:
        - Uses the query analysis to optimize retrieval
        - Performs similarity search in Pinecone
        - Returns ranked documents with scores
        
        Args:
            analysis: Query analysis from the first agent
            
        Returns:
            List of retrieved documents with metadata
        """
        logger.info("📚 Retrieval Agent: Searching vector database...")
        
        query = analysis['original_query']
        k = analysis.get('retrieval_count', 5)
        
        # Perform similarity search with scores
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            retrieved_docs = []
            for i, (doc, score) in enumerate(results):
                # Convert distance to similarity (Pinecone uses cosine similarity)
                # Score is already a similarity score (higher is better)
                similarity_score = score
                
                doc_dict = {
                    'rank': i + 1,
                    'content': doc.page_content,
                    'score': float(similarity_score),
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'Unknown')
                }
                retrieved_docs.append(doc_dict)
            
            logger.info(f"   ✓ Retrieved {len(retrieved_docs)} documents")
            for i, doc in enumerate(retrieved_docs[:3]):
                logger.info(f"   ✓ Doc {i+1} score: {doc['score']:.4f}")
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"   ✗ Retrieval error: {e}")
            return []
    
    def reflection_agent(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Agent 3: Reflects on retrieved documents and decides if they're sufficient.
        
        This agent:
        - Evaluates quality of retrieved documents
        - Checks if documents answer the query
        - Decides if re-retrieval with modified query is needed
        - Provides confidence score
        
        Args:
            query: Original user query
            retrieved_docs: Documents retrieved by retrieval agent
            
        Returns:
            Dictionary with reflection results and decision
        """
        logger.info("🤔 Reflection Agent: Evaluating retrieved documents...")
        
        if not retrieved_docs:
            return {
                'sufficient': False,
                'confidence': 0.0,
                'reasoning': 'No documents retrieved',
                'action': 'expand_query'
            }
        
        # Create a summary of retrieved docs for evaluation
        docs_summary = "\n\n".join([
            f"Document {i+1} (Score: {doc['score']:.4f}):\n{doc['content'][:300]}..."
            for i, doc in enumerate(retrieved_docs[:3])
        ])
        
        prompt = f"""You are a reflection agent evaluating the quality of retrieved documents for a medical query.

User Query: {query}

Retrieved Documents:
{docs_summary}

Evaluate whether these documents can adequately answer the user's query. Consider:
1. Relevance: Do the documents address the query topic?
2. Completeness: Is there enough information to answer fully?
3. Quality: Are the documents authoritative and accurate?
4. Coverage: Do they cover different aspects if needed?

Provide your evaluation in this format:
SUFFICIENT: yes/no
CONFIDENCE: 0.0-1.0
REASONING: <explain your assessment>
ACTION: keep/expand_query/refine_query
SUGGESTED_REFINEMENT: <if action is expand/refine, suggest how>
"""
        
        response = self.reflection_llm.invoke([HumanMessage(content=prompt)])
        reflection_text = response.content
        
        # Parse reflection
        reflection = self._parse_reflection(reflection_text)
        
        logger.info(f"   ✓ Sufficient: {reflection.get('sufficient', False)}")
        logger.info(f"   ✓ Confidence: {reflection.get('confidence', 0.0):.2f}")
        logger.info(f"   ✓ Action: {reflection.get('action', 'keep')}")
        
        return reflection
    
    def _parse_reflection(self, text: str) -> Dict[str, Any]:
        """Parse the reflection response."""
        reflection = {}
        
        lines = text.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'sufficient':
                    reflection['sufficient'] = value.lower() in ['yes', 'true']
                elif key == 'confidence':
                    try:
                        reflection['confidence'] = float(value)
                    except:
                        reflection['confidence'] = 0.5
                elif key == 'reasoning':
                    reflection['reasoning'] = value
                elif key == 'action':
                    reflection['action'] = value
                elif key == 'suggested_refinement':
                    reflection['suggested_refinement'] = value
        
        # Set defaults
        if 'sufficient' not in reflection:
            reflection['sufficient'] = True
        if 'confidence' not in reflection:
            reflection['confidence'] = 0.5
        if 'action' not in reflection:
            reflection['action'] = 'keep'
            
        return reflection
    
    def response_synthesis_agent(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        reflection: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Agent 4: Synthesizes the final response based on retrieved documents.
        
        This agent:
        - Generates comprehensive answer from retrieved documents
        - Incorporates chat history for context
        - Provides citations and confidence scores
        - Uses step-by-step reasoning (Chain of Thought)
        
        Args:
            query: Original user query
            retrieved_docs: Retrieved documents
            reflection: Reflection agent's evaluation
            chat_history: Optional conversation history
            
        Returns:
            Dictionary with final response and metadata
        """
        logger.info("✍️  Response Synthesis Agent: Generating response...")
        
        # Format context from retrieved documents
        context = self._format_context(retrieved_docs)
        
        # Format chat history if provided
        history_text = ""
        if chat_history:
            history_text = "Previous conversation:\n"
            for msg in chat_history[-4:]:  # Last 2 exchanges
                history_text += f"- {msg.get('role', 'user')}: {msg.get('content', '')}\n"
            history_text += "\n"
        
        # Build prompt with Chain of Thought
        prompt = f"""You are a compassionate and professional medical doctor with extensive experience in patient care and medical education. Your responses should be:

🌟 **CORE PRINCIPLES:**
- **Polite & Empathetic**: Always speak with warmth, kindness, and genuine concern for the patient's wellbeing
- **Elaborate & Thorough**: Provide comprehensive, detailed explanations that educate and empower the patient
- **Simple & Clear**: Use everyday language while maintaining medical accuracy - avoid jargon or explain it simply
- **Professional & Trustworthy**: Speak as a caring physician would - confident yet humble, informative yet cautious
- **Patient-Centered**: Focus on the individual's health journey, concerns, and needs

{history_text}

User Question: {query}

Retrieved Medical Knowledge:
{context}

Retrieval Confidence: {reflection.get('confidence', 0.5):.2f}

Instructions:
1. **Think Step-by-Step**: Break down the question and reason through it
2. **Use Retrieved Context**: Answer ONLY based on the provided medical knowledge
3. **Be Accurate**: If information is insufficient, acknowledge limitations
4. **Cite Sources**: Reference document numbers when using specific information
5. **Professional Tone**: Use clear, professional medical language

If the retrieved context doesn't fully answer the question, state: "Based on the available information, I can tell you that..." and provide what you can.

Now, let's think through this step by step:

Step 1: Understand the Question
- What is the user asking?
- What key information do they need?

Step 2: Analyze Retrieved Information
- What relevant information is in the documents?
- Is it sufficient to answer the question?

Step 3: Formulate Response
- How can I best answer based on available information?
- What limitations should I note?

Step 4: Provide Final Answer

Your Response (as a caring physician):"""
        
        response = self.synthesis_llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content
        
        # Extract sources
        sources = [
            {
                'title': doc.get('source', f"Document {doc['rank']}"),
                'score': doc['score'],
                'rank': doc['rank']
            }
            for doc in retrieved_docs[:5]
        ]
        
        logger.info("   ✓ Response generated successfully")
        
        return {
            'response': response_text,
            'sources': sources,
            'confidence': reflection.get('confidence', 0.5),
            'retrieved_doc_count': len(retrieved_docs)
        }
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context for the LLM."""
        context = ""
        for doc in documents:
            context += f"\n\n=== Document {doc['rank']} (Relevance: {doc['score']:.4f}) ===\n"
            context += doc['content']
            context += f"\nSource: {doc.get('source', 'Unknown')}"
        return context
    
    def query(
        self,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_reflection_iterations: int = 2
    ) -> Dict[str, Any]:
        """
        Main agentic RAG workflow that orchestrates all agents.
        
        Workflow:
        1. Query Analysis Agent analyzes the query
        2. Retrieval Agent retrieves relevant documents
        3. Reflection Agent evaluates retrieval quality
        4. (Optional) Re-retrieve with refined query if needed
        5. Response Synthesis Agent generates final answer
        
        Args:
            user_query: User's question
            chat_history: Optional conversation history
            max_reflection_iterations: Maximum times to refine and re-retrieve
            
        Returns:
            Final response dictionary with answer and metadata
        """
        logger.info("\n" + "="*80)
        logger.info(f"🎯 New Query: {user_query}")
        logger.info("="*80)
        
        # Step 1: Query Analysis
        analysis = self.query_analysis_agent(user_query)
        
        # Step 2: Initial Retrieval
        retrieved_docs = self.retrieval_agent(analysis)
        
        # Step 3: Reflection Loop (with max iterations)
        iteration = 0
        while iteration < max_reflection_iterations:
            reflection = self.reflection_agent(user_query, retrieved_docs)
            
            # If documents are sufficient or we've reached max iterations, break
            if reflection['sufficient'] or iteration == max_reflection_iterations - 1:
                break
            
            # If not sufficient, refine query and re-retrieve
            if reflection.get('action') in ['expand_query', 'refine_query']:
                logger.info(f"🔄 Iteration {iteration + 1}: Refining query and re-retrieving...")
                
                # Modify retrieval count or strategy
                analysis['retrieval_count'] = min(analysis['retrieval_count'] + 2, 10)
                retrieved_docs = self.retrieval_agent(analysis)
                
            iteration += 1
        
        # Step 4: Response Synthesis
        final_response = self.response_synthesis_agent(
            user_query,
            retrieved_docs,
            reflection,
            chat_history
        )
        
        # Add metadata
        final_response['query_analysis'] = analysis
        final_response['reflection'] = reflection
        final_response['iterations'] = iteration
        
        logger.info("="*80)
        logger.info("✅ Query processing complete!")
        logger.info("="*80 + "\n")
        
        return final_response


def demo_rag_only():
    """
    Demo script to test the Agentic RAG system only (backward compatibility).
    """
    print("\n" + "="*80)
    print("🏥 AGENTIC RAG SYSTEM DEMO - Medical Assistant (RAG Only)")
    print("="*80 + "\n")
    
    # Configuration - REPLACE WITH YOUR CREDENTIALS
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medagentica")
    OPENROUTER_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY"))
    OPENROUTER_MODEL = os.getenv("GROQ_MODEL", os.getenv("OPENROUTER_MODEL", "llama-3.3-70b-versatile"))
    
    # Check if credentials are set
    if PINECONE_API_KEY == "YOUR_PINECONE_API_KEY":
        print("❌ Error: Please set PINECONE_API_KEY environment variable")
        print("   You can set it in your .env file or export it:")
        print("   export PINECONE_API_KEY='your_api_key_here'")
        return
    
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
        print("❌ Error: Please set OPENROUTER_API_KEY environment variable")
        print("   You can set it in your .env file or export it:")
        print("   export OPENROUTER_API_KEY='your_api_key_here'")
        return
    
    try:
        # Initialize Agentic RAG System
        rag_system = AgenticRAGSystem(
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_index_name=PINECONE_INDEX_NAME,
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_model=OPENROUTER_MODEL
        )
        
        print("\n" + "="*80)
        print("📋 DEMO QUERIES")
        print("="*80 + "\n")
        
        # Demo Query 1: Simple factual query
        query1 = "What are the common symptoms of type 2 diabetes?"
        print(f"\n🔍 Query 1: {query1}\n")
        response1 = rag_system.query(query1)
        
        print("\n📝 Response:")
        print(response1['response'])
        print(f"\n📊 Confidence: {response1['confidence']:.2%}")
        print(f"📚 Documents Retrieved: {response1['retrieved_doc_count']}")
        print(f"🔄 Reflection Iterations: {response1['iterations']}")
        
        # Demo Query 2: Complex medical query
        print("\n" + "-"*80 + "\n")
        query2 = "How does metformin work in managing blood glucose levels?"
        print(f"\n🔍 Query 2: {query2}\n")
        response2 = rag_system.query(query2, chat_history=[
            {'role': 'user', 'content': query1},
            {'role': 'assistant', 'content': response1['response'][:200] + '...'}
        ])
        
        print("\n📝 Response:")
        print(response2['response'])
        print(f"\n📊 Confidence: {response2['confidence']:.2%}")
        print(f"📚 Documents Retrieved: {response2['retrieved_doc_count']}")
        print(f"🔄 Reflection Iterations: {response2['iterations']}")
        
        # Demo Query 3: Medical diagnosis query
        print("\n" + "-"*80 + "\n")
        query3 = "What are the treatment options for hypertension?"
        print(f"\n🔍 Query 3: {query3}\n")
        response3 = rag_system.query(query3)
        
        print("\n📝 Response:")
        print(response3['response'])
        print(f"\n📊 Confidence: {response3['confidence']:.2%}")
        print(f"📚 Documents Retrieved: {response3['retrieved_doc_count']}")
        print(f"🔄 Reflection Iterations: {response3['iterations']}")
        
        print("\n" + "="*80)
        print("✅ Demo completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def demo_full_system():
    """
    Comprehensive demo script to test the full Multi-Agent Medical Assistant system.
    This includes all agents: RAG, Web Search, Brain Tumor, Chest X-ray (with MedRAX), Skin Lesion, and Conversation.
    """
    print("\n" + "="*80)
    print("🏥 MULTI-AGENT MEDICAL ASSISTANT - Full System Demo")
    print("="*80 + "\n")
    
    # Import the full agent system
    try:
        from agents.agent_decision import process_query
        print("✅ Successfully imported full agent system")
    except Exception as e:
        print(f"❌ Error importing agent system: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("📋 DEMO QUERIES - Testing All Agents")
    print("="*80 + "\n")
    
    try:
        # Demo Query 1: RAG Agent - Medical knowledge query
        print("\n" + "="*80)
        print("🔍 Query 1: RAG Agent - Medical Knowledge")
        print("="*80)
        query1 = "What are the common symptoms of type 2 diabetes?"
        print(f"\nUser Query: {query1}\n")
        
        response_data1 = process_query(query1)
        response1 = response_data1['messages'][-1].content if response_data1.get('messages') else str(response_data1)
        agent1 = response_data1.get('agent_name', 'Unknown')
        
        print(f"\n🤖 Selected Agent: {agent1}")
        print(f"\n📝 Response:")
        print(response1)
        print("\n" + "-"*80)
        
        # Demo Query 2: Web Search Agent - Recent medical information
        print("\n" + "="*80)
        print("🔍 Query 2: Web Search Agent - Recent Medical Information")
        print("="*80)
        query2 = "What are the latest treatments for COVID-19 in 2024?"
        print(f"\nUser Query: {query2}\n")
        
        response_data2 = process_query(query2)
        response2 = response_data2['messages'][-1].content if response_data2.get('messages') else str(response_data2)
        agent2 = response_data2.get('agent_name', 'Unknown')
        
        print(f"\n🤖 Selected Agent: {agent2}")
        print(f"\n📝 Response:")
        print(response2[:500] + "..." if len(response2) > 500 else response2)
        print("\n" + "-"*80)
        
        # Demo Query 3: Conversation Agent - General chat
        print("\n" + "="*80)
        print("🔍 Query 3: Conversation Agent - General Chat")
        print("="*80)
        query3 = "Hello! How are you?"
        print(f"\nUser Query: {query3}\n")
        
        response_data3 = process_query(query3)
        response3 = response_data3['messages'][-1].content if response_data3.get('messages') else str(response_data3)
        agent3 = response_data3.get('agent_name', 'Unknown')
        
        print(f"\n🤖 Selected Agent: {agent3}")
        print(f"\n📝 Response:")
        print(response3)
        print("\n" + "-"*80)
        
        # Demo Query 4: Chest X-ray Agent (with MedRAX) - Image analysis
        print("\n" + "="*80)
        print("🔍 Query 4: Chest X-ray Agent (MedRAX) - Image Analysis")
        print("="*80)
        
        # Try to find a sample chest X-ray image
        import pathlib
        project_root = pathlib.Path(__file__).parent.absolute()
        sample_images_dir = project_root / "sample_images" / "chest_x-ray_covid_and_normal"
        
        chest_xray_image = None
        if sample_images_dir.exists():
            image_files = list(sample_images_dir.glob("*.jpg")) + list(sample_images_dir.glob("*.jpeg")) + list(sample_images_dir.glob("*.png"))
            if image_files:
                chest_xray_image = str(image_files[0])
                print(f"\n✅ Found sample image: {chest_xray_image}")
        
        if chest_xray_image and os.path.exists(chest_xray_image):
            query4 = "Do I have TB or not? Analyze this chest X-ray."
            print(f"\nUser Query: {query4}")
            print(f"Image: {chest_xray_image}\n")
            
            response_data4 = process_query({
                "text": query4,
                "image": chest_xray_image
            })
            response4 = response_data4['messages'][-1].content if response_data4.get('messages') else str(response_data4)
            agent4 = response_data4.get('agent_name', 'Unknown')
            
            print(f"\n🤖 Selected Agent: {agent4}")
            print(f"\n📝 Response:")
            print(response4[:1000] + "..." if len(response4) > 1000 else response4)
            
            # Show image URLs if available
            if response_data4.get('original_image_url'):
                print(f"\n🖼️  Original Image: {response_data4.get('original_image_url')}")
            if response_data4.get('segmentation_image_url'):
                print(f"🖼️  Segmentation Image: {response_data4.get('segmentation_image_url')}")
            if response_data4.get('disease_grounding_url'):
                print(f"🖼️  Disease Grounding Image: {response_data4.get('disease_grounding_url')}")
        else:
            print("\n⚠️  Note: No sample chest X-ray image found.")
            print("   To test this, provide an image path in the query dictionary.")
            print("   Example: process_query({'text': 'Analyze this chest X-ray', 'image': '/path/to/image.jpg'})")
        
        print("\n" + "-"*80)
        
        # Demo Query 5: Brain Tumor Agent - Image analysis
        print("\n" + "="*80)
        print("🔍 Query 5: Brain Tumor Agent - Image Analysis")
        print("="*80)
        print("\n⚠️  Note: This requires a brain MRI image file.")
        print("   To test this, provide an image path in the query dictionary.")
        print("   Example: process_query({'text': 'Analyze this brain MRI', 'image': '/path/to/image.jpg'})")
        print("\n" + "-"*80)
        
        # Demo Query 6: Skin Lesion Agent - Image analysis
        print("\n" + "="*80)
        print("🔍 Query 6: Skin Lesion Agent - Image Analysis")
        print("="*80)
        print("\n⚠️  Note: This requires a skin lesion image file.")
        print("   To test this, provide an image path in the query dictionary.")
        print("   Example: process_query({'text': 'Analyze this skin lesion', 'image': '/path/to/image.jpg'})")
        print("\n" + "-"*80)
        
        # Update summary based on what was tested
        chest_xray_tested = chest_xray_image and os.path.exists(chest_xray_image) if 'chest_xray_image' in locals() else False
        
        print("\n" + "="*80)
        print("✅ Full System Demo completed successfully!")
        print("="*80)
        print("\n📝 Summary:")
        print(f"   • RAG Agent: ✅ Tested")
        print(f"   • Web Search Agent: ✅ Tested")
        print(f"   • Conversation Agent: ✅ Tested")
        if chest_xray_tested:
            print(f"   • Chest X-ray Agent (MedRAX): ✅ Tested with sample image")
            print(f"     - 18-disease classification")
            print(f"     - Anatomical segmentation")
            print(f"     - Report generation (Findings & Impression)")
            print(f"     - Disease grounding/visualization")
        else:
            print(f"   • Chest X-ray Agent (MedRAX): ⚠️  Requires image file")
        print(f"   • Brain Tumor Agent: ⚠️  Requires image file")
        print(f"   • Skin Lesion Agent: ⚠️  Requires image file")
        print("\n💡 To test image agents, use:")
        print("   process_query({'text': 'Your query', 'image': '/path/to/image.jpg'})")
        print("\n💡 Chest X-ray Agent Features (MedRAX):")
        print("   - Multi-disease classification (18 pathologies)")
        print("   - Anatomical structure segmentation")
        print("   - Automated radiology report generation")
        print("   - Disease localization with bounding boxes")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main demo function - runs the full system demo by default.
    Use --rag-only flag to run only the RAG demo.
    """
    import sys
    
    if "--rag-only" in sys.argv:
        demo_rag_only()
    else:
        demo_full_system()


if __name__ == "__main__":
    main()



