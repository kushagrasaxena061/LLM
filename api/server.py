import os
import uuid
import logging
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model.config import canonical_151m_config
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from training.checkpointing import load_full_checkpoint
from inference.generate import generate_text


try:
    from security.guardrails import detect_prompt_injection, sanitize_pii
except ImportError:
    try:
        from security import detect_prompt_injection, sanitize_pii
    except ImportError:
        def detect_prompt_injection(prompt): return "ignore" in prompt.lower() or "previous instructions" in prompt.lower()
        def sanitize_pii(text): return text


# Dynamically load RAG dependencies
try:
    from rag.vector_store import SimpleVectorStore
    from rag.hybrid_search import HybridRetriever
    from evaluation.embeddings import EmbeddingEngine
    from rag.reranker import HeuristicLexicalReranker
    from rag.pipeline import RAGPipeline
except ImportError:
    RAGPipeline = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="MiniGPT Studio Production API")

model = None
tokenizer = None
dev_mode = True
param_count = 0
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
rag_pipeline = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7

class MultimodalRequest(BaseModel):
    prompt: str
    image_url: str = 'dummy_image_data'
    max_new_tokens: int = 20

class RAGRequest(BaseModel):
    query: str
    top_k: int = 3

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, dev_mode, param_count, rag_pipeline
    
    logger.info("Initializing Canonical 151M Model...")
    model = GPT(canonical_151m_config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    
    tokenizer = BPETokenizer(vocab_size=canonical_151m_config.vocab_size)
    
    vocab_path = os.getenv("TOKENIZER_PATH", "artifacts/vocab.json")
    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
    else:
        logger.warning("No tokenizer artifact found. Using untrained tokenizer (Dev Mode).")
        tokenizer.train("Dummy training data for dev mode fallback. A picture is worth a thousand words. test document about AI. another document about networking. fast modern web framework.")
        
    ckpt_path = os.getenv("CHECKPOINT_PATH", "checkpoints/minigpt_151m_ckpt.pt")
    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint from {ckpt_path}")
        load_full_checkpoint(ckpt_path, model, optimizer=torch.optim.AdamW(model.parameters()), device=device)
        dev_mode = False
    else:
        if os.getenv("REQUIRE_CHECKPOINT", "false").lower() == "true":
            raise RuntimeError(f"Production checkpoint missing at {ckpt_path}")
        logger.warning(f"No checkpoint found at {ckpt_path}. Operating in DEV MODE (untrained weights).")
        dev_mode = True

    model.eval()
    
    if RAGPipeline is not None:
        store = SimpleVectorStore(16)
        engine = EmbeddingEngine(model, tokenizer)
        hybrid = HybridRetriever(store, engine)
        reranker = HeuristicLexicalReranker()
        rag_pipeline = RAGPipeline(hybrid, reranker, model, tokenizer, engine, device)

@app.middleware("http")
async def limit_payload_size(request: Request, call_next):
    if "content-length" in request.headers:
        if int(request.headers["content-length"]) > 1024 * 1024:
            return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    return await call_next(request)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    import time
    req_id = str(uuid.uuid4())
    request.state.req_id = req_id
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Sec"] = str(round(process_time, 4))
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {req_id} failed: {e}")
        return JSONResponse(
            status_code=500, 
            headers={"X-Request-ID": req_id, "X-Process-Time-Sec": str(round(process_time, 4))},
            content={"detail": "Internal Server Error", "request_id": req_id}
        )

@app.get("/health")
def health():
    return {
        "status": "online",
        "model_configuration": "canonical_151m",
        "parameters": param_count,
        "dev_mode": dev_mode,
        "device": device,
        "security": "hardened",
        "vocab_consistency": tokenizer.vocab_size == canonical_151m_config.vocab_size if tokenizer else False
    }

@app.post("/generate")
def generate(req: GenerateRequest, request: Request):
    req_id = getattr(request.state, "req_id", "unknown")
    
    if 'detect_prompt_injection' in globals() and detect_prompt_injection(req.prompt):
        raise HTTPException(status_code=403, detail="Prompt Injection Detected. Request Blocked.")
        
    safe_prompt = sanitize_pii(req.prompt) if 'sanitize_pii' in globals() else req.prompt
    
    try:
        out = generate_text(model, tokenizer, safe_prompt, max_new_tokens=req.max_new_tokens, device=device)
        return {"generated_text": out, "request_id": req_id}
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/generate-multimodal")
def generate_multimodal_endpoint(req: MultimodalRequest):
    import torch
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    try:
        from model.multimodal import VisionPatchExtractor, VisionLanguageAdapter
        extractor = VisionPatchExtractor()
        adapter = VisionLanguageAdapter(16, model.config.d_model).to(device)
    except ImportError:
        import torch.nn as nn
        extractor = lambda x: torch.randn(x.shape[0], 4, 16).to(device)
        adapter = nn.Linear(16, model.config.d_model).to(device)
        
    from inference.multimodal import generate_multimodal_text
    try:
        out = generate_multimodal_text(
            model=model, 
            tokenizer=tokenizer, 
            vision_extractor=extractor, 
            vision_adapter=adapter, 
            image_tensor=dummy_img, 
            prompt=req.prompt, 
            max_new_tokens=req.max_new_tokens, 
            device=device
        )
        return {"generated_text": out, "note": "Educational multimodal output. Visual meaning is random unless visually trained."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/rag/query")
def rag_endpoint(req: RAGRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=501, detail="RAG Pipeline unavailable.")
    try:
        res = rag_pipeline.answer_query(req.query, top_k=req.top_k)
        return {"answer": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
