import math
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# 2. Prepare the knowledge base
documents = [
    "Employees receive 15 days of paid annual leave per year.",
    "Remote work requires approval from the direct supervisor.",
    "Employees can apply for annual leave after working for six months.",
    "The company provides health insurance to all full-time employees.",
]


# 3. Generate document embeddings
document_embeddings = model.encode(documents)


# 4. Prepare the user query
query = "When can I apply for annual leave?"

query_embedding = model.encode(query)


# 5. Calculate cosine similarity
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

# 6. Define the retrieval function
def retrieve(query, documents, document_embeddings, model, top_k=2):
    # Generate query embedding
    query_embedding = model.encode(query)

    # Normalize query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Calculate cosine similarities
    scores = document_embeddings @ query_embedding

    # Pair each score with its document
    results = list(zip(scores, documents))

    # Rank documents by similarity
    results.sort(key=lambda x: x[0], reverse=True)

    # Return Top-K results
    return results[:top_k]


# 7. Normalize document embeddings
document_embeddings = document_embeddings / np.linalg.norm(
    document_embeddings,
    axis=1,
    keepdims=True
)


# 8. Retrieve relevant documents
query = "When can I apply for annual leave?"

top_results = retrieve(
    query=query,
    documents=documents,
    document_embeddings=document_embeddings,
    model=model,
    top_k=2
)


# 9. Display results
print(f"Query: {query}")
print(f"\nTop {len(top_results)} Results:")

for rank, (score, document) in enumerate(top_results, start=1):
    print(f"\nRank: {rank}")
    print(f"Similarity: {score:.4f}")
    print(f"Document: {document}")
