import math

documents = [
    "Employees receive 15 days of paid annual leave per year.",
    "Remote work requires approval from the direct supervisor.",
    "Employees can apply for annual leave after working for six months.",
    "The company provides health insurance to all full-time employees.",
]

embeddings = [
    [0.1, 0.2, 0.3],
    [0.9, 0.1, 0.2],
    [0.8, 0.7, 0.6],
    [0.2, 0.1, 0.4],
]

query_embedding = [0.7, 0.8, 0.6]


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    return dot_product / (norm_a * norm_b)

results = []
for document, embedding in zip(documents, embeddings):
    score = cosine_similarity(query_embedding, embedding)

    results.append((score, document))

results.sort(key=lambda x: x[0], reverse=True)

top_k = 2

top_results = results[:top_k]

print(f"Top 2 Results:")
for score, document in top_results:
    print(f"{score}, {document}")