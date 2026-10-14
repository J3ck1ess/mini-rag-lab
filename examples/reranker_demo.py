from sentence_transformers import CrossEncoder

from mini_rag import chunks, document_embeddings, model, retrieve


# 1. Load the Cross-Encoder model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 2. Define a query that previously failed
query = (
    "After what period of employment can staff "
    "request annual leave?"
)


# 3. Retrieve candidate chunks using the embedding model
top_k = 5

retrieved_results = retrieve(
    query=query,
    chunks=chunks,
    document_embeddings=document_embeddings,
    model=model,
    top_k=top_k,
)


print("=== Before Reranking ===")

for rank, (chunk_id, score, chunk) in enumerate(retrieved_results, start=1):
    print(f"Rank: {rank}")
    print(f"Chunk ID: {chunk_id}")
    print(f"Embedding Score: {score:.4f}")
    print(f"Content:\n{chunk}")
    print("-" * 50)


# 4. Prepare query-document pairs for the Cross-Encoder
pairs = [
    (query, chunk)
    for _, _, chunk in retrieved_results
]


# 5. Score each query-document pair
reranker_scores = reranker.predict(pairs)


# 6. Attach the new scores to the original results
reranked_results = [
    (chunk_id, float(reranker_score), chunk)
    for (chunk_id, _, chunk), reranker_score
    in zip(retrieved_results, reranker_scores)
]


# 7. Sort by Cross-Encoder score in descending order
reranked_results.sort(
    key=lambda item: item[1],
    reverse=True,
)


print("\n=== After Reranking ===")

for rank, (chunk_id, score, chunk) in enumerate(reranked_results, start=1):
    print(f"Rank: {rank}")
    print(f"Chunk ID: {chunk_id}")
    print(f"Cross-Encoder Score: {score:.4f}")
    print(f"Content:\n{chunk}")
    print("-" * 50)


print("\n=== Cross-Encoder Score Diagnosis ===")
print(f"Query: {query}")
print(f"Relevant Chunk IDs: [1]")

for rank, (chunk_id, score, chunk) in enumerate(
    reranked_results,
    start=1,
):
    print(f"\nRank: {rank}")
    print(f"Chunk ID: {chunk_id}")
    print(f"Cross-Encoder Score: {score:.4f}")
    print(f"Content:\n{chunk}")