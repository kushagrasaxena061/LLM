# frontend/app.py
"""Interactive Streamlit dashboard for the custom LLM & RAG platform."""

# frontend/app.py
"""Interactive Streamlit dashboard for the custom LLM & RAG platform."""

import sys
from pathlib import Path

# Automatically add the project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import streamlit as st
import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.generate import generate_text
from rag.vector_store import SimpleVectorStore
from rag.pipeline import RAGPipeline


import streamlit as st
import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.generate import generate_text
from rag.vector_store import SimpleVectorStore
from rag.pipeline import RAGPipeline

# Page Config
st.set_page_config(
    page_title="Custom LLM & RAG Platform",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_platform_components():
    """Initializes and caches model, tokenizer, and RAG pipeline for the UI."""
    config = GPTConfig(vocab_size=260, context_length=64, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config).to(env_config.device)
    model.eval()
    
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("The quick brown fox jumps over the lazy dog. Streamlit brings Python apps to life.")
    
    vector_store = SimpleVectorStore(embedding_dim=32)
    docs = [
        "Streamlit is an open-source Python library that makes it easy to build beautiful web apps.",
        "Retrieval-Augmented Generation enhances LLM accuracy by fetching external context."
    ]
    torch.manual_seed(42)
    embeddings = torch.randn(2, 32, device=env_config.device)
    vector_store.add_texts(docs, embeddings)
    
    rag_pipeline = RAGPipeline(vector_store, model, tokenizer, env_config.device)
    return model, tokenizer, rag_pipeline

model, tokenizer, rag_pipeline = load_platform_components()

# UI Layout
st.title("🤖 Custom LLM & RAG Studio")
st.markdown(f"**Hardware Device Active:** `{env_config.device.upper()}` | **Architecture:** Decoder-Only GPT with LoRA, INT8 Quantization, & RAG")

tab1, tab2, tab3 = st.tabs(["✨ Text Generation", "📚 RAG Query", "📊 System Analytics"])

with tab1:
    st.subheader("Autoregressive Text Generation")
    prompt = st.text_input("Enter your prompt:", value="The quick")
    max_tokens = st.slider("Max New Tokens", 5, 50, 20)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
    
    if st.button("Generate Response"):
        with st.spinner("Generating tokens..."):
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                device=env_config.device,
                temperature=temperature
            )
            st.success("Generation Complete!")
            st.text_area("Output:", value=output, height=100)

with tab2:
    st.subheader("Retrieval-Augmented Generation (RAG)")
    query = st.text_input("Ask a question based on local vector database docs:", value="What is Streamlit?")
    
    if st.button("Search & Answer"):
        with st.spinner("Retrieving context and generating answer..."):
            torch.manual_seed(42)
            query_embedding = torch.randn(32, device=env_config.device)
            response = rag_pipeline.answer_query(query, query_embedding, top_k=1, max_new_tokens=25)
            st.success("RAG Pipeline Executed!")
            st.markdown(response)

with tab3:
    st.subheader("Model & System Statistics")
    st.json({
        "Total Parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        "Embedding Dimension": model.config.d_model,
        "Transformer Layers": model.config.n_layers,
        "Attention Heads": model.config.n_heads,
        "Context Length": model.config.context_length,
        "Running Device": env_config.device
    })
