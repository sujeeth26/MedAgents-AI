"""
Demo Script to Ingest Medical Documents into Pinecone Vector Database

This script:
1. Loads PDF documents from a directory
2. Extracts text and tables using pdfplumber
3. Chunks the documents intelligently
4. Generates embeddings using HuggingFace
5. Uploads to Pinecone vector database

Usage:
    python demo_ingest_pinecone.py
"""

import os
import pdfplumber
from typing import List, Dict, Any
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PineconeDataIngestion:
    """
    Handles ingestion of medical documents into Pinecone vector database.
    """
    
    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str = "medagentica",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            pinecone_api_key: Pinecone API key
            index_name: Name of the Pinecone index
            embedding_model: HuggingFace embedding model name
        """
        logger.info("🚀 Initializing Pinecone Data Ingestion Pipeline...")
        
        self.index_name = index_name
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize embeddings
        logger.info(f"📊 Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Ensure index exists
        self._ensure_index_exists()
        
        logger.info("✅ Initialization complete!")
    
    def _ensure_index_exists(self):
        """
        Ensure the Pinecone index exists, create if it doesn't.
        """
        logger.info(f"🔍 Checking if index '{self.index_name}' exists...")
        
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"📦 Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # sentence-transformers/all-MiniLM-L6-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info("✅ Index created successfully!")
            # Wait for index to be ready
            time.sleep(5)
        else:
            logger.info(f"✅ Index '{self.index_name}' already exists")
    
    def extract_text_and_tables_from_pdf(self, pdf_path: str) -> str:
        """
        Extract both text and tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Combined text content with tables
        """
        logger.info(f"📄 Extracting content from: {os.path.basename(pdf_path)}")
        
        full_text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    # Extract regular text
                    page_text = page.extract_text() or ""
                    full_text += f"\n\n[Page {i}]:\n{page_text}"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for j, table in enumerate(tables, start=1):
                            # Convert table to string
                            table_str = "\n".join(["\t".join(str(cell) for cell in row) for row in table if row])
                            full_text += f"\n\n[Page {i} - Table {j}]:\n{table_str}"
            
            logger.info(f"   ✓ Extracted {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"   ✗ Error extracting from PDF: {e}")
            # Fallback to basic PyPDF loader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return "\n\n".join([doc.page_content for doc in documents])
    
    def chunk_documents(
        self,
        text: str,
        source: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Chunk the document text into smaller pieces for better retrieval.
        
        Args:
            text: Full text content
            source: Source file name
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of Document objects
        """
        logger.info(f"✂️  Chunking document: {source}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create a single document
        doc = Document(page_content=text, metadata={"source": source})
        
        # Split into chunks
        chunks = splitter.split_documents([doc])
        
        logger.info(f"   ✓ Created {len(chunks)} chunks")
        return chunks
    
    def ingest_pdf(
        self,
        pdf_path: str,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Ingest a single PDF file into Pinecone.
        
        Args:
            pdf_path: Path to the PDF file
            batch_size: Batch size for uploading to Pinecone
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📥 Ingesting: {os.path.basename(pdf_path)}")
        logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Step 1: Extract text and tables
            full_text = self.extract_text_and_tables_from_pdf(pdf_path)
            
            # Step 2: Chunk the document
            chunks = self.chunk_documents(full_text, os.path.basename(pdf_path))
            
            # Step 3: Upload to Pinecone in batches
            logger.info("📤 Uploading to Pinecone...")
            
            total_chunks = len(chunks)
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]
                
                # Create/update vectorstore with this batch
                vectorstore = PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    index_name=self.index_name
                )
                
                logger.info(f"   ✓ Uploaded batch {i // batch_size + 1}: {i + 1}-{min(i + batch_size, total_chunks)} of {total_chunks}")
            
            processing_time = time.time() - start_time
            
            logger.info(f"\n✅ Successfully ingested {os.path.basename(pdf_path)}")
            logger.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
            
            return {
                "success": True,
                "file": os.path.basename(pdf_path),
                "chunks": total_chunks,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error ingesting {os.path.basename(pdf_path)}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "file": os.path.basename(pdf_path),
                "error": str(e)
            }
    
    def ingest_directory(
        self,
        directory_path: str,
        file_pattern: str = "*.pdf"
    ) -> Dict[str, Any]:
        """
        Ingest all PDF files from a directory into Pinecone.
        
        Args:
            directory_path: Path to directory containing PDF files
            file_pattern: Pattern for file matching
            
        Returns:
            Dictionary with overall statistics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📂 Ingesting directory: {directory_path}")
        logger.info(f"{'='*80}\n")
        
        # Get all PDF files
        pdf_files = list(Path(directory_path).glob(file_pattern))
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {directory_path}")
            return {
                "success": False,
                "error": "No PDF files found"
            }
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files to ingest\n")
        
        results = []
        total_start_time = time.time()
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n📖 Processing file {i}/{len(pdf_files)}")
            result = self.ingest_pdf(str(pdf_path))
            results.append(result)
        
        total_time = time.time() - total_start_time
        
        # Calculate statistics
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_chunks = sum(r.get('chunks', 0) for r in results if r['success'])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 INGESTION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"✅ Successful: {successful}/{len(pdf_files)}")
        logger.info(f"❌ Failed: {failed}/{len(pdf_files)}")
        logger.info(f"📄 Total chunks ingested: {total_chunks}")
        logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
        logger.info(f"{'='*80}\n")
        
        return {
            "success": True,
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "total_chunks": total_chunks,
            "processing_time": total_time,
            "results": results
        }


def main():
    """
    Main function to run the ingestion demo.
    """
    print("\n" + "="*80)
    print("📥 PINECONE DATA INGESTION DEMO")
    print("="*80 + "\n")
    
    # Configuration - REPLACE WITH YOUR CREDENTIALS
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medagentica")
    
    # Path to your PDF files - REPLACE WITH YOUR PATH
    PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./data/raw")
    
    # Check if credentials are set
    if PINECONE_API_KEY == "YOUR_PINECONE_API_KEY":
        print("❌ Error: Please set PINECONE_API_KEY environment variable")
        print("   You can set it in your .env file or export it:")
        print("   export PINECONE_API_KEY='your_api_key_here'")
        return
    
    # Check if directory exists
    if not os.path.exists(PDF_DIRECTORY):
        print(f"❌ Error: Directory not found: {PDF_DIRECTORY}")
        print("   Please set PDF_DIRECTORY environment variable to your PDF folder:")
        print("   export PDF_DIRECTORY='/path/to/your/pdfs'")
        return
    
    try:
        # Initialize ingestion pipeline
        ingestion = PineconeDataIngestion(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME
        )
        
        # Ingest all PDFs from directory
        results = ingestion.ingest_directory(PDF_DIRECTORY)
        
        if results['success']:
            print("\n✅ Ingestion completed successfully!")
            print(f"📊 Ingested {results['total_chunks']} chunks from {results['successful']} files")
        else:
            print("\n❌ Ingestion failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



