import os

api_path = 'api/server.py'
with open(api_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Skip the old, late initialization at the bottom
    if "embedding_engine = EmbeddingEngine(model)" in line:
        continue
    # Inject it right before the RAG vector store needs it!
    if "vector_store = SimpleVectorStore(embedding_dim=32)" in line:
        new_lines.append("    embedding_engine = EmbeddingEngine(model)\n")
    new_lines.append(line)

with open(api_path, 'w') as f:
    f.writelines(new_lines)

print("\n✅ API Server Startup Order Fixed! Real RAG embeddings will now initialize correctly.")
