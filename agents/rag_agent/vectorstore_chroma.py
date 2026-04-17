import os
import logging
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain.storage import LocalFileStore

class ChromaVectorStore:
    """
    ChromaDB-based vector store for RAG system
    """
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.rag.collection_name
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        self.chroma_persist_directory = "./data/chroma_db"
        self.docstore_local_path = config.rag.doc_local_path

    def load_vectorstore(self) -> Tuple[Chroma, LocalFileStore]:
        """
        Load existing ChromaDB vectorstore and docstore for retrieval operations.
        
        Returns:
            Tuple containing (vectorstore, docstore)
        """
        try:
            # Initialize ChromaDB vectorstore
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.chroma_persist_directory
            )
            
            # Document storage
            docstore = LocalFileStore(self.docstore_local_path)
            
            self.logger.info(f"Successfully loaded ChromaDB vectorstore and docstore")
            return vectorstore, docstore
            
        except Exception as e:
            self.logger.error(f"Error loading ChromaDB vectorstore: {e}")
            raise e

    def retrieve_relevant_chunks(
            self,
            query: str,
            vectorstore: Chroma,
            docstore: LocalFileStore,
        ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks using ChromaDB similarity search.
        
        Args:
            query: Search query
            vectorstore: ChromaDB vectorstore instance
            docstore: Document store instance
            
        Returns:
            List of relevant document dictionaries
        """
        try:
            # Perform similarity search with scores
            results = vectorstore.similarity_search_with_score(query, k=self.retrieval_top_k)
            
            retrieved_docs = []
            
            for doc, score in results:
                # ChromaDB returns distance (lower is better), convert to similarity (higher is better)
                similarity_score = 1.0 / (1.0 + score) if score > 0 else 1.0
                
                # Create document dict in the format expected by the response generator
                doc_dict = {
                    "id": doc.metadata.get('doc_id', f'chroma_doc_{len(retrieved_docs)}'),
                    "content": doc.page_content,
                    "score": similarity_score,
                    "source": doc.metadata.get('source', 'Medical Knowledge Base'),
                    "source_path": doc.metadata.get('source_path', 'http://localhost:8000/data/chroma_db'),
                }
                retrieved_docs.append(doc_dict)
            
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents from ChromaDB")
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents from ChromaDB: {e}")
            return []
