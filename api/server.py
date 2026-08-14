# api/server.py
"""Hardened FastAPI backend for MiniGPT Studio."""

import time
import uuid
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from model.config import canonical_151m_config
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.generate import generate_text
from rag.vector_store import SimpleVectorStore
from rag.pipeline import RAGPipeline
from security.guardrails import SecurityGuard

app = FastAPI(title="MiniGPT Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(canonical_151m_config).to(device)
model.eval()

tokenizer = BPETokenizer(vocab_size=canonical_151m_config.vocab_size)
security_guard = SecurityGuard()

# Mock RAG pipeline for API
vector_store = SimpleVectorStore(embedding_dim=canonical_151m_config.d_model)
rag_pipeline = RAGPipeline(vector_store, model, tokenizer, device)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=4000)
    max_new_tokens: int = Field(default=50, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class RAGRequest(BaseModel):
    query: str = Field(..., max_length=4000)
    max_new_tokens: int = Field(default=50, le=512)

@app.middleware("http")
async def security_and_telemetry_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1_000_000:
        return JSONResponse(status_code=413, content={"detail": "Payload Too Large"})

    response = await call_next(request)
    
    process_time = time.perf_counter() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Sec"] = str(process_time)
    return response

def sanitize_pii(text: str) -> str:
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\d{3}-\d{2}-\d{4}', '[REDACTED_PHONE]', text)
    return text

@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    sec = security_guard.validate_input(req.prompt)
    if not sec["is_safe"]:
        return JSONResponse(status_code=403, content={"detail": "Prompt Injection Detected."})
    try:
        output = generate_text(
            model=model, tokenizer=tokenizer, prompt=sec["sanitized_prompt"],
            max_new_tokens=req.max_new_tokens, device=device, temperature=req.temperature,
            stop_tokens=["<|im_end|>", "<|endoftext|>"]
        )
        return {"response": sanitize_pii(output)}
    except Exception:
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

@app.post("/rag/query")
async def rag_query_endpoint(req: RAGRequest):
    sec = security_guard.validate_input(req.query)
    if not sec["is_safe"]:
        return JSONResponse(status_code=403, content={"detail": "Prompt Injection Detected."})
    try:
        query_embedding = torch.randn(canonical_151m_config.d_model, device=device)
        output = rag_pipeline.answer_query(
            query=sec["sanitized_prompt"], query_embedding=query_embedding,
            top_k=1, max_new_tokens=req.max_new_tokens
        )
        return {"rag_response": sanitize_pii(output)}
    except Exception:
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

@app.get("/health")
async def health_check():
    return {"status": "online", "model_parameters": model.get_num_params(), "device": device, "security": "hardened"}
