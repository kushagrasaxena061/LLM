# frontend/app.py
"""Interactive Streamlit dashboard for the custom LLM & RAG platform."""

import sys
from pathlib import Path
from PIL import Image

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
from prompt_engineering.optimizer import PromptOptimizer
from multimodal.vision_adapter import VisionLanguageAdapter, combine_embeddings
from personas.engine import PersonaManager
from explainability.visualizer import AttentionVisualizer

# Page Config
st.set_page_config(
    page_title="Custom LLM & RAG Platform",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_platform_components():
    """Initializes and caches all platform components."""
    config = GPTConfig(vocab_size=260, context_length=256, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config).to(env_config.device)
    model.eval()
    
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("The quick brown fox jumps over the lazy dog. Streamlit brings Python apps to life. please could you kindly help me write a python script to parse json Describe this image:")
    
    vector_store = SimpleVectorStore(embedding_dim=32)
    docs = [
        "Streamlit is an open-source Python library that makes it easy to build beautiful web apps.",
        "Retrieval-Augmented Generation enhances LLM accuracy by fetching external context."
    ]
    torch.manual_seed(42)
    embeddings = torch.randn(2, 32, device=env_config.device)
    vector_store.add_texts(docs, embeddings)
    
    rag_pipeline = RAGPipeline(vector_store, model, tokenizer, env_config.device)
    optimizer = PromptOptimizer(tokenizer)
    vision_adapter = VisionLanguageAdapter(vision_dim=512, llm_dim=32).to(env_config.device)
    persona_manager = PersonaManager()
    visualizer = AttentionVisualizer(model, tokenizer, env_config.device)
    
    return model, tokenizer, rag_pipeline, optimizer, vision_adapter, persona_manager, visualizer

model, tokenizer, rag_pipeline, optimizer, vision_adapter, persona_manager, visualizer = load_platform_components()

# UI Layout
st.title("🤖 Custom LLM & RAG Studio")
st.markdown(f"**Hardware Device Active:** `{env_config.device.upper()}` | **Architecture:** GPT + LoRA + INT8 + RAG + Vision + Personas")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "✨ Text Gen", 
    "📚 RAG Query", 
    "📊 Analytics", 
    "🛠️ Optimizer", 
    "👁️ Multimodal",
    "🎭 Personas",
    "🔍 Transformer Explorer"
])

with tab1:
    st.subheader("Autoregressive Text Generation")
    prompt = st.text_input("Enter your prompt:", value="The quick")
    max_tokens = st.slider("Max New Tokens", 5, 50, 20)
    if st.button("Generate Response"):
        with st.spinner("Generating tokens..."):
            output = generate_text(model, tokenizer, prompt, max_tokens, env_config.device)
            st.success("Generation Complete!")
            st.text_area("Output:", value=output, height=100)

with tab2:
    st.subheader("Retrieval-Augmented Generation (RAG)")
    query = st.text_input("Ask a question based on local vector database docs:", value="What is Streamlit?")
    if st.button("Search & Answer"):
        with st.spinner("Retrieving context..."):
            torch.manual_seed(42)
            query_embedding = torch.randn(32, device=env_config.device)
            response = rag_pipeline.answer_query(query, query_embedding, top_k=1, max_new_tokens=25)
            st.success("RAG Executed!")
            st.markdown(response)

with tab3:
    st.subheader("Model & System Statistics")
    st.json({
        "Total Parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        "Embedding Dimension": model.config.d_model,
        "Context Length": model.config.context_length,
        "Running Device": env_config.device
    })

with tab4:
    st.subheader("Prompt Optimization Engine")
    raw_prompt = st.text_area("Enter raw user prompt:", value="Please could you kindly help me write a python script to parse json?")
    if st.button("Optimize Prompt"):
        result = optimizer.optimize_prompt(raw_prompt)
        col1, col2 = st.columns(2)
        col1.metric("Original Tokens", result["original_tokens"])
        col2.metric("Optimized Tokens", result["optimized_tokens"], delta=f"-{result['tokens_saved']} tokens", delta_color="inverse")
        st.write(f"**Optimized Prompt:** {result['optimized_prompt']}")

with tab5:
    st.subheader("Multimodal Vision-Language Projection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        if st.button("Process Multimodal Sequence"):
            with st.spinner("Extracting vision patches and projecting..."):
                torch.manual_seed(42)
                vision_tensor = torch.randn(1, 16, 512, device=env_config.device)
                projected_vision = vision_adapter(vision_tensor)
                prompt_text = "Describe this image:"
                text_tokens = torch.tensor([tokenizer.encode(prompt_text)], device=env_config.device)
                text_embeddings = model.tok_embeddings(text_tokens)
                combined = combine_embeddings(text_embeddings, projected_vision)
                
                st.success("Multimodal Sequence Assembled!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Raw Vision Tensor", f"{list(vision_tensor.shape)}")
                col2.metric("Projected Tensor", f"{list(projected_vision.shape)}")
                col3.metric("Final Fusion Sequence", f"{list(combined.shape)}")

with tab6:
    st.subheader("Persona Studio & Context Control")
    selected_persona = st.selectbox("Select Active Persona:", list(persona_manager.presets.keys()))
    persona_data = persona_manager.get_persona(selected_persona)
    col1, col2 = st.columns(2)
    col1.info(f"**System Prompt:**\n{persona_data.system_prompt}")
    col2.metric("Enforced Temperature", persona_data.temperature)
    
    st.divider()
    test_user_prompt = st.text_input("Simulate User Input:", value="Write a function to sort an array.")
    if st.button("Preview ChatML Injection"):
        formatted_prompt = persona_manager.apply_persona(test_user_prompt, selected_persona)
        st.success("Payload formatted for the LLM Engine:")
        st.text_area("Under-the-hood ChatML Array:", value=formatted_prompt, height=150)

with tab7:
    st.subheader("Transformer Explorer & Attention Heatmaps")
    st.markdown("Peer inside the mathematical black box. See exactly which tokens the model focuses on during generation.")
    
    explore_text = st.text_input("Enter a short phrase to analyze:", value="The quick brown fox jumps.")
    head_to_view = st.slider("Select Attention Head:", min_value=0, max_value=model.config.n_heads - 1, value=0)
    
    if st.button("Generate Attention Heatmap"):
        with st.spinner("Extracting weights from Layer 0..."):
            tokens, attn_matrix = visualizer.extract_attention(explore_text)
            fig = visualizer.plot_attention_heatmap(tokens, attn_matrix, head_idx=head_to_view)
            
            st.success("Attention Matrix Extracted!")
            st.pyplot(fig)
            
            with st.expander("How to read this heatmap?"):
                st.write("The **Y-axis (Query)** represents the current token being processed.")
                st.write("The **X-axis (Key)** represents the historical tokens it is 'looking at'.")
                st.write("Brighter colors mean higher mathematical attention (closer to **1.0**). Notice how tokens cannot look at future tokens due to Causal Masking!")
