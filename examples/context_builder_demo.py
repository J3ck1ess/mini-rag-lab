from mini_rag import retrieve, chunks, document_embeddings, model
from reranker import rerank
from context_builder import build_context
from sentence_transformers import CrossEncoder


query = "How many months must an employee work before qualifying for annual leave?"

# Step 1: Retrieve candidate chunks
candidates = retrieve(
    query,
    chunks,
    document_embeddings,
    model,
    top_k=5
)

# Step 2: Rerank candidates
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked_results = rerank(query, candidates, reranker_model)

# Step 3: Build context
context = build_context(reranked_results, max_chunks=3)

print("Query:")
print(query)

print("\nBuilt Context:")
print(context)