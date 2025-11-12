"""
FastAPI RAG Application for HR Assistant
"""
import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HR Assistant RAG API",
    description="Retrieval-Augmented Generation API for HR Policy Questions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class Source(BaseModel):
    page: int
    text: str
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Source]
    processing_time: float

# Global clients (initialized on startup)
azure_client = None
pinecone_index = None

@app.on_event("startup")
async def startup_event():
    """Initialize clients on startup"""
    global azure_client, pinecone_index
    
    logger.info("="*60)
    logger.info("Starting HR Assistant RAG API")
    logger.info("="*60)
    
    try:
        # Initialize Azure OpenAI
        logger.info("Initializing Azure OpenAI client...")
        azure_client = AzureOpenAI(
            api_key=os.getenv("HR_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("HR_AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("HR_AZURE_OPENAI_ENDPOINT")
        )
        logger.info(f"✅ Azure OpenAI initialized with deployment: {os.getenv('HR_AZURE_OPENAI_DEPLOYMENT_NAME')}")
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pinecone_index = pc.Index(os.getenv("INDEX_NAME"))
        
        # Check index stats
        stats = pinecone_index.describe_index_stats()
        logger.info(f"✅ Pinecone connected to index: {os.getenv('INDEX_NAME')}")
        logger.info(f"   Index stats: {stats}")
        
        logger.info("="*60)
        logger.info("✅ API Ready to serve requests")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HR Assistant RAG API",
        "status": "active",
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Pinecone connection
        stats = pinecone_index.describe_index_stats()
        
        return {
            "status": "healthy",
            "azure_openai": "connected",
            "pinecone": "connected",
            "vectors_count": stats.get('total_vector_count', 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Process user query using RAG (Retrieval-Augmented Generation)
    
    1. Embed the query
    2. Retrieve relevant chunks from Pinecone
    3. Generate answer using GPT-4 with context
    """
    start_time = time.time()
    logger.info("="*60)
    logger.info(f"📥 Received query: {request.query}")
    logger.info("="*60)
    
    try:
        # Step 1: Generate query embedding
        logger.info("[STEP 1] Generating query embedding...")
        embedding_start = time.time()
        
        embedding_response = azure_client.embeddings.create(
            input=request.query,
            model=os.getenv("HR_AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            dimensions=512  # Match Pinecone index dimension
        )
        query_embedding = embedding_response.data[0].embedding
        
        embedding_time = time.time() - embedding_start
        logger.info(f"✅ Query embedded in {embedding_time:.2f}s")
        
        # Step 2: Retrieve relevant chunks from Pinecone
        logger.info(f"[STEP 2] Querying Pinecone for top {request.top_k} results...")
        retrieval_start = time.time()
        
        search_results = pinecone_index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True
        )
        
        retrieval_time = time.time() - retrieval_start
        logger.info(f"✅ Retrieved {len(search_results.matches)} results in {retrieval_time:.2f}s")
        
        # Extract context from results
        sources = []
        context_parts = []
        
        for i, match in enumerate(search_results.matches, 1):
            metadata = match.metadata
            score = match.score
            
            sources.append(Source(
                page=metadata.get('page', 0),
                text=metadata.get('text', '')[:200] + "...",  # Preview only
                score=round(score, 4)
            ))
            
            context_parts.append(f"[Source {i} - Page {metadata.get('page')}]:\n{metadata.get('text', '')}")
            logger.info(f"   Match {i}: Page {metadata.get('page')}, Score: {score:.4f}")
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Generate answer using GPT-4
        logger.info("[STEP 3] Generating answer with GPT-4...")
        generation_start = time.time()
        
        system_prompt = """You are an HR assistant helping employees with questions about company policies. 
Use the provided context from the HR Policy Manual to answer questions accurately and helpfully.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite the page number when providing information."""
        
        user_prompt = f"""Context from HR Policy Manual:
{context}

Employee Question: {request.query}

Please provide a clear, accurate answer based on the context above."""
        
        completion = azure_client.chat.completions.create(
            model=os.getenv("HR_AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        answer = completion.choices[0].message.content
        generation_time = time.time() - generation_start
        
        logger.info(f"✅ Answer generated in {generation_time:.2f}s")
        logger.info(f"   Tokens used - Prompt: {completion.usage.prompt_tokens}, Completion: {completion.usage.completion_tokens}")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        logger.info("="*60)
        logger.info(f"✅ Query processed successfully in {total_time:.2f}s")
        logger.info(f"   Breakdown - Embedding: {embedding_time:.2f}s, Retrieval: {retrieval_time:.2f}s, Generation: {generation_time:.2f}s")
        logger.info("="*60)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            processing_time=round(total_time, 2)
        )
        
    except Exception as e:
        logger.error("="*60)
        logger.error(f"❌ Error processing query: {str(e)}")
        logger.error("="*60)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = pinecone_index.describe_index_stats()
        return {
            "index_name": os.getenv("INDEX_NAME"),
            "total_vectors": stats.get('total_vector_count', 0),
            "dimension": stats.get('dimension', 0),
            "index_fullness": stats.get('index_fullness', 0)
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
