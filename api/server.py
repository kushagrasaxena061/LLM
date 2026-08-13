# api/server.py
"""Production-grade FastAPI server for LLM inference and RAG services."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from contextlib import asynccontextmanager

from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.generate import generate_text
from rag.vector_store import SimpleVectorStore
from rag.pipeline import RAGPipeline
from security.guardrails import SecurityGuard
from observability.middleware import TelemetryMiddleware
from utils.logger import get_logger

logger = get_logger(__name__)

model = None
tokenizer = None
rag_pipeline = None
security_guard = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, rag_pipeline, security_guard
    logger.info("Initializing API server components...")
    
    config = GPTConfig(vocab_size=260, context_length=256, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config).to(env_config.device)
    model.eval()
    
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("The quick brown fox jumps over the lazy dog. FastAPI is a modern web framework.")
    
    vector_store = SimpleVectorStore(embedding_dim=32)
    docs = ["FastAPI is a modern, fast web framework for building APIs with Python."]
    torch.manual_seed(42)
    embeddings = torch.randn(1, 32, device=env_config.device)
    vector_store.add_texts(docs, embeddings)
    
    rag_pipeline = RAGPipeline(vector_store, model, tokenizer, env_config.device)
    security_guard = SecurityGuard()
    
    logger.info("API server startup complete.")
    yield

app = FastAPI(
    title="Custom LLM & RAG Platform API",
    version="1.0.0",
    description="Production backend serving a custom decoder-only transformer with LoRA, Quantization, and RAG.",
    lifespan=lifespan
)

# Inject the Observability Middleware
app.add_middleware(TelemetryMiddleware)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt for generation")
    max_new_tokens: int = Field(default=20, ge=1, le=100, description="Number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")

class RAGRequest(BaseModel):
    query: str = Field(..., description="User question for RAG retrieval")
    max_new_tokens: int = Field(default=20, description="Number of response tokens")

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": env_config.device}

@app.post("/generate")
def generate(request: GenerateRequest):
    security_check = security_guard.validate_input(request.prompt)
    if not security_check["is_safe"]:
        logger.warning("API Blocked Malicious Request", pattern=security_check["matched_pattern"])
        raise HTTPException(status_code=403, detail="Prompt Injection Detected. Request Blocked.")
        
    safe_prompt = security_check["sanitized_prompt"]
    
    try:
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=safe_prompt,
            max_new_tokens=request.max_new_tokens,
            device=env_config.device,
            temperature=request.temperature
        )
        return {"prompt": safe_prompt, "generated_text": output}
    except Exception as e:
        logger.error("Generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
def rag_query(request: RAGRequest):
    security_check = security_guard.validate_input(request.query)
    if not security_check["is_safe"]:
        logger.warning("API Blocked Malicious RAG Request")
        raise HTTPException(status_code=403, detail="Prompt Injection Detected. Request Blocked.")
        
    safe_query = security_check["sanitized_prompt"]

    try:
        torch.manual_seed(42)
        query_embedding = torch.randn(32, device=env_config.device)
        
        output = rag_pipeline.answer_query(
            query=safe_query,
            query_embedding=query_embedding,
            top_k=1,
            max_new_tokens=request.max_new_tokens
        )
        return {"query": safe_query, "rag_response": output}
    except Exception as e:
        logger.error("RAG query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
