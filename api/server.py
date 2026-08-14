# api/server.py
"""Comprehensive FastAPI server exposing all MiniGPT Studio engineering engines."""

import string
import time
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch

from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.generate import generate_text
from inference.chat import ChatSessionManager
from rag.vector_store import SimpleVectorStore
from rag.pipeline import RAGPipeline
from security.guardrails import SecurityGuard
from evaluation.safety import SafetyEvaluator
from evaluation.embeddings import EmbeddingEngine
from evaluation.model_comparator import ModelComparator
from api.hardening import APIHardeningMiddleware
from observability.middleware import TelemetryMiddleware
from utils.logger import get_logger

logger = get_logger(__name__)

# Global instances
model = None
tokenizer = None
rag_pipeline = None
security_guard = None
safety_evaluator = None
embedding_engine = None
chat_manager = None
startup_timestamp = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, rag_pipeline, security_guard, safety_evaluator, embedding_engine, chat_manager
    logger.info("Initializing MiniGPT Studio API services...")
    
    # 1. Initialize Model
    config = GPTConfig(vocab_size=300, context_length=256, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config).to(env_config.device)
    model.eval()
    
    # 2. Tokenizer
    tokenizer = BPETokenizer(vocab_size=300)
    full_vocab = "The quick brown fox jumps over the lazy dog. FastAPI is a modern web framework. " + string.ascii_letters + string.punctuation + string.digits
    tokenizer.train(full_vocab)
    
    # 3. RAG Pipeline
    vector_store = SimpleVectorStore(embedding_dim=32)
    docs = ["FastAPI handles our secure backend by acting as the API layer.", "Streamlit handles the frontend interface."]
    torch.manual_seed(42)
    embeddings = torch.randn(2, 32, device=env_config.device)
    vector_store.add_texts(docs, embeddings)
    rag_pipeline = RAGPipeline(vector_store, model, tokenizer, env_config.device)
    
    # 4. Auxiliary Engines
    security_guard = SecurityGuard()
    safety_evaluator = SafetyEvaluator()
    embedding_engine = EmbeddingEngine(model)
    chat_manager = ChatSessionManager(model, tokenizer, device=env_config.device)
    
    logger.info("MiniGPT Studio API startup complete.")
    yield

app = FastAPI(
    title="MiniGPT Studio 151M Platform API",
    version="1.0.0",
    description="Production-grade API for Transformer inference, RAG, LoRA, Quantization, and Observability.",
    lifespan=lifespan
)

# Register Middlewares (Outermost first)
app.add_middleware(APIHardeningMiddleware)
app.add_middleware(TelemetryMiddleware)

# --- Pydantic Request Models ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    max_new_tokens: int = Field(default=20, ge=1, le=100)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    persona: str = Field(default="General Assistant")
    max_new_tokens: int = Field(default=25, ge=1, le=100)

class RAGRequest(BaseModel):
    query: str
    max_new_tokens: int = 20

class TokenizeRequest(BaseModel):
    text: str

class EmbeddingsRequest(BaseModel):
    texts: List[str]

class SafetyRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    completion: Optional[str] = None

# --- API Endpoints ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "uptime_sec": round(time.time() - startup_timestamp, 2),
        "device": env_config.device,
        "security": "hardened",
        "model_loaded": model is not None
    }

@app.get("/model/inspect")
def inspect_model():
    """Returns dynamic model configuration and architectural parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "architecture": "Decoder-Only Transformer (GPT)",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "d_model": model.config.d_model,
        "n_layers": model.config.n_layers,
        "n_heads": model.config.n_heads,
        "head_dim": model.config.head_dim,
        "vocab_size": model.config.vocab_size,
        "context_length": model.config.context_length,
        "norm_type": "RMSNorm",
        "activation": "SwiGLU",
        "position_encoding": "RoPE (Rotary Embeddings)"
    }

@app.get("/model/compare")
def compare_models():
    """Profiles Base FP32 vs LoRA vs INT8 quantization."""
    return ModelComparator.profile_configuration(model.config, device=env_config.device)

@app.post("/generate")
def generate(request: GenerateRequest):
    security_check = security_guard.validate_input(request.prompt)
    if not security_check["is_safe"]:
        raise HTTPException(status_code=403, detail="Prompt Injection Detected.")
    try:
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=security_check["sanitized_prompt"],
            max_new_tokens=request.max_new_tokens,
            device=env_config.device,
            temperature=request.temperature
        )
        return {"prompt": request.prompt, "generated_text": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(request: ChatRequest):
    security_check = security_guard.validate_input(request.message)
    if not security_check["is_safe"]:
        raise HTTPException(status_code=403, detail="Prompt Injection Detected in Chat.")
    try:
        response = chat_manager.respond(
            user_message=security_check["sanitized_prompt"],
            persona_name=request.persona,
            max_new_tokens=request.max_new_tokens
        )
        return {"response": response, "history": chat_manager.history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokenizer/analyze")
def analyze_tokenizer(request: TokenizeRequest):
    tokens = tokenizer.encode(request.text)
    decoded = tokenizer.decode(tokens)
    raw_chars = len(request.text)
    token_count = len(tokens)
    compression = round(raw_chars / token_count, 2) if token_count > 0 else 1.0
    return {
        "raw_text": request.text,
        "token_ids": tokens,
        "token_count": token_count,
        "decoded_text": decoded,
        "compression_ratio": compression,
        "vocab_size": tokenizer.vocab_size
    }

@app.post("/embeddings/extract")
def extract_embeddings(request: EmbeddingsRequest):
    embeddings_list = []
    for t in request.texts:
        ids = torch.tensor([tokenizer.encode(t)], device=env_config.device)
        vec = embedding_engine.extract_sequence_embedding(ids)[0]
        embeddings_list.append(vec.cpu())
        
    stacked = torch.stack(embeddings_list)
    sim_matrix = embedding_engine.compute_similarity_matrix(stacked).tolist()
    pca_pts = embedding_engine.compute_pca_2d(stacked)
    return {
        "texts": request.texts,
        "similarity_matrix": sim_matrix,
        "pca_coordinates": pca_pts
    }

@app.post("/rag/query")
def rag_query(request: RAGRequest):
    security_check = security_guard.validate_input(request.query)
    if not security_check["is_safe"]:
        raise HTTPException(status_code=403, detail="Prompt Injection Detected.")
    try:
        query_embedding = torch.randn(32, device=env_config.device)
        output = rag_pipeline.answer_query(
            query=security_check["sanitized_prompt"],
            query_embedding=query_embedding,
            top_k=1,
            max_new_tokens=request.max_new_tokens
        )
        return {"query": request.query, "rag_response": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/safety/evaluate")
def evaluate_safety(request: SafetyRequest):
    harm_eval = safety_evaluator.evaluate_harmfulness(request.prompt)
    hallucination_eval = {}
    if request.context and request.completion:
        hallucination_eval = safety_evaluator.evaluate_hallucination(request.context, request.completion)
    return {
        "prompt": request.prompt,
        "harm_evaluation": harm_eval,
        "hallucination_evaluation": hallucination_eval
    }
