"""
Script to chunk HR Policy PDF and upload embeddings to Pinecone
"""
import os
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor with Azure OpenAI and Pinecone clients"""
        logger.info("Initializing DocumentProcessor...")
        
        # Azure OpenAI setup
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("HR_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("HR_AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("HR_AZURE_OPENAI_ENDPOINT")
        )
        self.embedding_deployment = os.getenv("HR_AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        logger.info(f"Azure OpenAI client initialized with deployment: {self.embedding_deployment}")
        
        # Pinecone setup
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("INDEX_NAME")
        logger.info(f"Pinecone client initialized for index: {self.index_name}")
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page numbers"""
        logger.info(f"Reading PDF from: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            logger.info(f"PDF loaded successfully. Total pages: {total_pages}")
            
            pages_data = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    pages_data.append({
                        'page': page_num,
                        'text': text.strip()
                    })
                    logger.debug(f"Extracted {len(text)} characters from page {page_num}")
            
            logger.info(f"Successfully extracted text from {len(pages_data)} pages")
            return pages_data
        
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise
    
    def chunk_text(self, pages_data: List[Dict], chunk_size: int = 800, overlap: int = 200) -> List[Dict]:
        """Split text into overlapping chunks with metadata"""
        logger.info(f"Chunking text with size={chunk_size}, overlap={overlap}")
        
        chunks = []
        chunk_id = 0
        
        for page_data in pages_data:
            page_num = page_data['page']
            text = page_data['text']
            
            # Split into sentences (simple approach)
            words = text.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if chunk_text.strip():
                    chunks.append({
                        'id': f"chunk_{chunk_id}",
                        'text': chunk_text,
                        'page': page_num,
                        'source': 'HR Policy Manual 2023'
                    })
                    chunk_id += 1
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for chunks using Azure OpenAI"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        batch_size = 100
        embedded_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk['text'] for chunk in batch]
            
            try:
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                response = self.azure_client.embeddings.create(
                    input=texts,
                    model=self.embedding_deployment,
                    dimensions=512  # Match Pinecone index dimension
                )
                
                for j, chunk in enumerate(batch):
                    chunk['embedding'] = response.data[j].embedding
                    embedded_chunks.append(chunk)
                
                logger.info(f"Successfully embedded {len(batch)} chunks")
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
                raise
        
        logger.info(f"Total embeddings generated: {len(embedded_chunks)}")
        return embedded_chunks
    
    def upload_to_pinecone(self, embedded_chunks: List[Dict]):
        """Upload embedded chunks to Pinecone"""
        logger.info("Connecting to Pinecone index...")
        
        try:
            # Connect to index (assumes it already exists)
            index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to index: {self.index_name}")
            
            # Get index stats
            stats = index.describe_index_stats()
            logger.info(f"Current index stats: {stats}")
            
            # Prepare vectors for upsert
            vectors = []
            for chunk in embedded_chunks:
                vectors.append({
                    'id': chunk['id'],
                    'values': chunk['embedding'],
                    'metadata': {
                        'text': chunk['text'][:1000],  # Limit metadata size
                        'page': chunk['page'],
                        'source': chunk['source']
                    }
                })
            
            # Upsert in batches
            batch_size = 100
            logger.info(f"Uploading {len(vectors)} vectors to Pinecone...")
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1} ({len(batch)} vectors)")
                time.sleep(0.5)  # Rate limiting
            
            # Verify upload
            final_stats = index.describe_index_stats()
            logger.info(f"Upload complete! Final index stats: {final_stats}")
            
        except Exception as e:
            logger.error(f"Error uploading to Pinecone: {str(e)}")
            raise

def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("Starting HR Document Processing Pipeline")
    logger.info("="*60)
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        
        # Define PDF path
        pdf_path = Path(__file__).parent / "KB" / "HR Policy Manual 2023 (8).pdf"
        logger.info(f"PDF path: {pdf_path}")
        
        # Step 1: Extract text from PDF
        logger.info("\n[STEP 1] Extracting text from PDF...")
        pages_data = processor.extract_text_from_pdf(str(pdf_path))
        
        # Step 2: Chunk the text
        logger.info("\n[STEP 2] Chunking text...")
        chunks = processor.chunk_text(pages_data)
        
        # Step 3: Generate embeddings
        logger.info("\n[STEP 3] Generating embeddings...")
        embedded_chunks = processor.generate_embeddings(chunks)
        
        # Step 4: Upload to Pinecone
        logger.info("\n[STEP 4] Uploading to Pinecone...")
        processor.upload_to_pinecone(embedded_chunks)
        
        logger.info("\n" + "="*60)
        logger.info("✅ Pipeline completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error("\n" + "="*60)
        logger.error(f"❌ Pipeline failed: {str(e)}")
        logger.error("="*60)
        raise

if __name__ == "__main__":
    main()
