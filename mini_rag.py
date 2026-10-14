import re
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Prepare the knowledge base
document = """
Chapter 1: Annual Leave

Employees receive 15 days of paid annual leave per year.
Employees can apply for annual leave after working for six months.
Leave applications must be submitted at least 3 working days in advance.
All leave applications require approval from the direct supervisor.

Chapter 2: Remote Work

Remote work requires approval from the direct supervisor.
Employees must submit remote work requests through the company portal.

Chapter 3: Health Insurance

The company provides health insurance to all full-time employees.
Insurance coverage begins after the probation period.
"""


# 3. Split the document into chunks
def chunk_document(text):
    chunks = []
    current_chapter = ""

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if re.match(r"Chapter \d+:", line):
            current_chapter = line
            continue

        chunk = f"{current_chapter}\n\n{line}"
        chunks.append(chunk)

    return chunks


chunks = chunk_document(document)

# 4. Generate embeddings for all chunks
document_embeddings = model.encode(chunks)

# Normalize embeddings for cosine similarity
document_embeddings = document_embeddings / np.linalg.norm(
    document_embeddings,
    axis=1,
    keepdims=True
)


# 5. Retrieve relevant chunks
def retrieve(query, chunks, document_embeddings, model, top_k=2):
    # Generate query embedding
    query_embedding = model.encode(query)

    # Normalize query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Calculate cosine similarities
    scores = document_embeddings @ query_embedding

    # Pair each score with the chunk ID and text
    results = [
        (idx, float(score), chunk)
        for idx, (score, chunk) in enumerate(zip(scores, chunks))
    ]

    # Sort by similarity in descending order
    results.sort(key=lambda item: item[1], reverse=True)

    # Return the Top-K results
    return results[:top_k]


# 6. Run demo only when this file is executed directly
if __name__ == "__main__":

    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding shape: {document_embeddings.shape}")

    # Display Chunk ID Mapping
    print("\n=== Chunk ID Mapping ===")

    for idx, chunk in enumerate(chunks):
        print(f"\nChunk ID: {idx}")
        print(chunk)

    # Query the knowledge base
    query = "Who must approve my annual leave application?"

    top_results = retrieve(
        query=query,
        chunks=chunks,
        document_embeddings=document_embeddings,
        model=model,
        top_k=2
    )

    # Display retrieval results
    print(f"\nQuery: {query}")
    print(f"\nTop {len(top_results)} Results:")

    for rank, (chunk_id, score, chunk) in enumerate(
            top_results, start=1
    ):
        print(f"\n--- Rank {rank} ---")
        print(f"Chunk ID: {chunk_id}")
        print(f"Similarity: {score:.4f}")
        print(chunk)
