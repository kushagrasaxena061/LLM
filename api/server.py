# api/server.py
"""Production-grade FastAPI server for MiniGPT Studio with all 24 feature endpoints."""

import string
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
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
from multimodal.vision_adapter import VisionLanguageAdapter
from api.hardening import APIHardeningMiddleware
from observability.middleware import TelemetryMiddleware
from utils.logger import get_logger

logger = get_logger(__name__)

model = None
tokenizer = None
rag_pipeline = None
security_guard = None
safety_evaluator = None
embedding_engine = None
chat_manager = None
vision_adapter = None
startup_timestamp = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, rag_pipeline, security_guard, safety_evaluator, embedding_engine, chat_manager, vision_adapter
    logger.info("Initializing MiniGPT Studio API services...")
    
    config = GPTConfig(vocab_size=300, context_length=256, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config).to(env_config.device)
    model.eval()
    
    tokenizer = BPETokenizer(vocab_size=300)
    full_vocab = "The quick brown fox jumps over the lazy dog. FastAPI is a modern web framework. " + string.ascii_letters + string.punctuation + string.digits
    tokenizer.train(full_vocab)
    
    embedding_engine = EmbeddingEngine(model)
    vector_store = SimpleVectorStore(embedding_dim=32)
    docs = ["FastAPI handles our secure backend by acting as the API layer.", "Streamlit handles the frontend interface."]
    torch.manual_seed(42)
    embeddings = torch.stack([embedding_engine.extract_sequence_embedding(torch.tensor([tokenizer.encode(d)], device=env_config.device))[0] for d in docs]).detach()
    vector_store.add_texts(docs, embeddings)
    rag_pipeline = RAGPipeline(vector_store, model, tokenizer, env_config.device)
    
    security_guard = SecurityGuard()
    safety_evaluator = SafetyEvaluator()
    chat_manager = ChatSessionManager(model, tokenizer, device=env_config.device)
    vision_adapter = VisionLanguageAdapter(vision_dim=512, llm_dim=32).to(env_config.device)
    
    logger.info("MiniGPT Studio API startup complete.")
    yield

app = FastAPI(
    title="MiniGPT Studio 151M Platform API",
    version="1.0.0",
    description="Production API with all architectural, security, and explainability endpoints.",
    lifespan=lifespan
)

app.add_middleware(APIHardeningMiddleware)
app.add_middleware(TelemetryMiddleware)

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

class AttentionRequest(BaseModel):
    text: str
    layer_idx: int = 0
    head_idx: int = 0

class SecurityInspectRequest(BaseModel):
    prompt: str

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
        "context_length": model.config.context_length
    }

@app.get("/model/compare")
def compare_models():
    return ModelComparator.profile_configuration(model.config, device=env_config.device)

@app.post("/generate")
def generate(request: GenerateRequest):
    security_check = security_guard.validate_input(request.prompt)
    if not security_check["is_safe"]:
        raise HTTPException(status_code=403, detail="Prompt Injection Detected.")
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=security_check["sanitized_prompt"],
        max_new_tokens=request.max_new_tokens,
        device=env_config.device,
        temperature=request.temperature
    )
    return {"prompt": request.prompt, "generated_text": output}

@app.post("/chat")
def chat(request: ChatRequest):
    security_check = security_guard.validate_input(request.message)
    if not security_check["is_safe"]:
        raise HTTPException(status_code=403, detail="Prompt Injection Detected in Chat.")
    response = chat_manager.respond(
        user_message=security_check["sanitized_prompt"],
        persona_name=request.persona,
        max_new_tokens=request.max_new_tokens
    )
    return {"response": response, "history": chat_manager.history}

@app.post("/transformer/attention")
def extract_attention(request: AttentionRequest):
    token_ids = tokenizer.encode(request.text)
    if not token_ids:
        token_ids = [0]
    idx = torch.tensor([token_ids], dtype=torch.long, device=env_config.device)
    
    with torch.no_grad():
        _, _, _, attentions = model(idx, return_attention=True)
        
    layer = min(request.layer_idx, len(attentions) - 1)
    head = min(request.head_idx, attentions[layer].shape[1] - 1)
    
    # [1, n_heads, T, T] -> [T, T]
    attn_matrix = attentions[layer][0, head].cpu().tolist()
    token_labels = [tokenizer.decode([t]) for t in token_ids]
    
    return {
        "tokens": token_labels,
        "token_ids": token_ids,
        "sequence_length": len(token_ids),
        "layer": layer,
        "head": head,
        "attention_matrix": attn_matrix
    }

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
    query_ids = torch.tensor([tokenizer.encode(security_check["sanitized_prompt"])], device=env_config.device)
    query_embedding = embedding_engine.extract_sequence_embedding(query_ids)[0].detach()
    output = rag_pipeline.answer_query(
        query=security_check["sanitized_prompt"],
        query_embedding=query_embedding,
        top_k=1,
        max_new_tokens=request.max_new_tokens
    )
    return {"query": request.query, "rag_response": output}

@app.post("/security/inspect")
def inspect_security(request: SecurityInspectRequest):
    sec_res = security_guard.validate_input(request.prompt)
    harm_res = safety_evaluator.evaluate_harmfulness(request.prompt)
    
    is_safe = sec_res["is_safe"] and not harm_res["is_harmful"]
    final_action = "ALLOW" if is_safe else ("BLOCK" if not sec_res["is_safe"] else "FLAG")
    
    return {
        "prompt": request.prompt,
        "sanitized_prompt": sec_res["sanitized_prompt"],
        "prompt_injection_detected": not sec_res["is_safe"],
        "harm_evaluation": harm_res,
        "final_action": final_action
    }
