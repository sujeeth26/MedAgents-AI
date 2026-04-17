"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
import chromadb

# Load environment variables from .env file
load_dotenv()

def create_llm(temperature=0.7, use_local=False):
    """
    Create LLM instance based on configuration.
    Supports Ollama (local), Groq, OpenRouter.ai, and OpenAI.
    
    Args:
        temperature: Temperature for generation
        use_local: If True, use local Ollama model (MedGemma)
    """
    # Check if local Ollama should be used (for most agents)
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "alibayram/medgemma:4b")
    
    if use_local or use_ollama:
        # Use local Ollama (MedGemma)
        print(f"🏠 Using local Ollama model: {ollama_model}")
        return ChatOpenAI(
            model=ollama_model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Ollama doesn't need a real API key
            temperature=temperature
        )
    
    # Check if Groq is configured (highest priority for API calls)
    groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    groq_model = os.getenv("GROQ_MODEL") or os.getenv("groq_model") or "llama-3.3-70b-versatile"
    
    if groq_api_key and groq_api_key.startswith("gsk_"):
        # Use Groq
        return ChatOpenAI(
            model=groq_model,
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature
        )
    
    # Check if OpenRouter is configured (try both uppercase and lowercase)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key")
    openrouter_model = os.getenv("OPENROUTER_MODEL") or os.getenv("openrouter_model") or "deepseek/deepseek-chat-v3.1:free"
    
    if openrouter_api_key and openrouter_api_key != "YOUR_OPENROUTER_API_KEY" and not openrouter_api_key.startswith("gsk_"):
        # Use OpenRouter.ai
        return ChatOpenAI(
            model=openrouter_model,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature
        )
    
    # Fallback to OpenAI (try both uppercase and lowercase)
    openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    if openai_api_key and openai_api_key != "YOUR_OPENAI_API_KEY" and not openai_api_key.startswith("gsk_"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL") or os.getenv("openai_model") or "gpt-4o",
            api_key=openai_api_key,
            temperature=temperature
        )
    
    # Use Azure OpenAI as fallback
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("azure_openai_api_key")
    if azure_api_key and azure_api_key != "YOUR_OPENAI_API_KEY":
        return AzureChatOpenAI(
            deployment_name=os.getenv("deployment_name"),
            model_name=os.getenv("model_name", "gpt-4o"),
            azure_endpoint=os.getenv("azure_endpoint"),
            openai_api_key=azure_api_key,
            openai_api_version=os.getenv("openai_api_version"),
            temperature=temperature
        )
    
    # No API keys found - provide helpful error message
    raise ValueError(
        "\n\n❌ No LLM API keys found!\n\n"
        "Please set one of the following in your .env file:\n"
        "  • USE_OLLAMA=true (use local MedGemma)\n"
        "  • GROQ_API_KEY (fast & free!)\n"
        "  • OPENROUTER_API_KEY\n"
        "  • OPENAI_API_KEY\n"
        "  • AZURE_OPENAI_API_KEY\n\n"
        "See demo_env_template.txt for reference.\n"
    )

def create_embedding_model():
    """
    Create embedding model based on configuration.
    Supports HuggingFace, OpenAI, and other providers.
    """
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    
    if embedding_provider == "huggingface":
        # Use HuggingFace embeddings (free and reliable)
        return HuggingFaceEmbeddings(
            model_name=os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
    elif embedding_provider == "openai":
        openai_embedding_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
        if openai_embedding_key and openai_embedding_key != "YOUR_OPENAI_API_KEY":
            return OpenAIEmbeddings(
                api_key=openai_embedding_key,
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            )
        else:
            # Fallback to Azure OpenAI embeddings
            return AzureOpenAIEmbeddings(
                deployment=os.getenv("embedding_deployment_name"),
                model=os.getenv("embedding_model_name", "text-embedding-ada-002"),
                azure_endpoint=os.getenv("embedding_azure_endpoint"),
                openai_api_key=os.getenv("embedding_openai_api_key"),
                openai_api_version=os.getenv("embedding_openai_api_version")
            )
    else:
        # Default to HuggingFace embeddings (free and reliable)
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

def create_vectorstore(embedding_model, collection_name="medical_assistance_rag"):
    """
    Create vector store based on configuration.
    Supports Pinecone, ChromaDB, FAISS, and Qdrant.
    """
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    vector_provider = os.getenv("VECTOR_STORE_PROVIDER", "pinecone").lower()
    
    if vector_provider == "pinecone":
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "medical-assistant-embeddings")
        
        # Get or create index
        try:
            existing_indexes = [index.name for index in pc.list_indexes()]
            if index_name not in existing_indexes:
                from pinecone import ServerlessSpec
                pc.create_index(
                    name=index_name,
                    dimension=384,  # sentence-transformers/all-MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
        except Exception as e:
            print(f"Warning: Could not create Pinecone index: {e}")
        
        # Get the index and create vectorstore
        index = pc.Index(index_name)
        return PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            text_key="text"
        )
    
    elif vector_provider == "chromadb":
        default_path = os.path.join(project_root, "data/chroma_db")
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", default_path)
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
    
    elif vector_provider == "faiss":
        default_path = os.path.join(project_root, "data/faiss_db")
        persist_directory = os.getenv("FAISS_PERSIST_DIRECTORY", default_path)
        return FAISS.load_local(
            persist_directory,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    
    else:
        # Default to ChromaDB
        default_path = os.path.join(project_root, "data/chroma_db")
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", default_path)
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )

class AgentDecisionConfig:
    def __init__(self):
        # Use Groq API for fast routing (10x faster than local)
        self.llm = create_llm(temperature=0.1, use_local=False)  # Deterministic, FAST

class ConversationConfig:
    def __init__(self):
        # Use Groq API for fast conversation (10x faster than local)
        self.llm = create_llm(temperature=0.7, use_local=False)  # Creative but factual, FAST

class WebSearchConfig:
    def __init__(self):
        # Use OpenRouter for web search (needs good summarization)
        self.llm = create_llm(temperature=0.3, use_local=False)  # Use API (OpenRouter/Groq)
        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        # Get the project root directory (where config.py is located)
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Vector database configuration
        self.vector_db_type = os.getenv("VECTOR_STORE_PROVIDER", "chromadb").lower()
        self.embedding_dim = 384  # sentence-transformers/all-MiniLM-L6-v2 dimension
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = os.path.join(project_root, "data/qdrant_db")  # Keep for Qdrant compatibility
        self.doc_local_path = os.path.join(project_root, "data/docs_db")
        self.parsed_content_dir = os.path.join(project_root, "data/parsed_docs")
        
        # Qdrant configuration (for backward compatibility)
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "medical_assistance_rag")
        
        # Document processing configuration
        self.chunk_size = 512
        self.chunk_overlap = 50
        
        # Initialize embedding model using the new helper function
        self.embedding_model = create_embedding_model()
        
        # Initialize LLM models using the new helper function
        # RAG uses Groq API (for better retrieval quality)
        self.llm = create_llm(temperature=0.3, use_local=False)  # Use API (Groq)
        self.summarizer_model = create_llm(temperature=0.5, use_local=False)  # Use API (Groq)
        self.chunker_model = create_llm(temperature=0.0, use_local=False)  # Use API (Groq)
        self.response_generator_model = create_llm(temperature=0.3, use_local=False)  # Use API (Groq)
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = None  # Disabled due to authentication issues
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.20  # Lower this if RAG is too restrictive
        
        # Add debug flag
        self.debug_mode = True  # Set to False in production
        
        self.context_limit = 20
    
    def get_vectorstore(self):
        """
        Get or create vector store based on current configuration.
        """
        return create_vectorstore(self.embedding_model, self.collection_name)

class MedicalCVConfig:
    def __init__(self):
        # Get the project root directory (where config.py is located)
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Use absolute paths to model files
        self.brain_tumor_model_path = os.path.join(project_root, "agents/image_analysis_agent/brain_tumor_agent/BrainMRI-Tumor-Classifier-Pytorch-main/models/model_38")
        self.chest_xray_model_path = os.path.join(project_root, "agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth")
        self.skin_lesion_model_path = os.path.join(project_root, "agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar")
        self.skin_lesion_segmentation_output_path = os.path.join(project_root, "uploads/skin_lesion_output/overlayed_plot.png")
        # Use remote LLM for medical image analysis to ensure reliability and better prompting
        self.llm = create_llm(temperature=0.1, use_local=False)  # Use API for reliable generation

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8001
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisionConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history

# # Example usage
# config = Config()