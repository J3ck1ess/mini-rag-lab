from mini_rag import chunks, document_embeddings, model, retrieve

# Each query maps to the IDs of all relevant chunks.
evaluation_dataset = [
    {
        "query": "How many days of annual leave do employees receive?",
        "relevant_chunk_ids": [0],
    },
    {
        "query": "When can I apply for annual leave?",
        "relevant_chunk_ids": [1],
    },
    {
        "query": "How long must I work before I can apply for annual leave?",
        "relevant_chunk_ids": [1],
    },
    {
        "query": "How many working days in advance must I submit my leave application?",
        "relevant_chunk_ids": [2],
    },
    {
        "query": "Who must approve my annual leave application?",
        "relevant_chunk_ids": [3],
    },
    {
        "query": "What are the requirements for annual leave?",
        "relevant_chunk_ids": [1, 2, 3],
    },
    {
        "query": "Who approves remote work requests?",
        "relevant_chunk_ids": [4],
    },
    {
        "query": "How do I submit a remote work request?",
        "relevant_chunk_ids": [5],
    },
    {
        "query": "Who receives health insurance from the company?",
        "relevant_chunk_ids": [6],
    },
    {
        "query": "When does health insurance coverage begin?",
        "relevant_chunk_ids": [7],
    },

    # Challenge queries: paraphrased and multi-condition questions

    {
        "query": (
            "What is the waiting period before an employee "
            "becomes eligible for paid leave?"
        ),
        "relevant_chunk_ids": [1],
    },
    {
        "query": (
            "How many business days ahead should a leave "
            "request be filed?"
        ),
        "relevant_chunk_ids": [2],
    },
    {
        "query": (
            "What do I need to do to work from home?"
        ),
        "relevant_chunk_ids": [4, 5],
    },
    {
        "query": (
            "Which employees are covered by the company's "
            "medical insurance plan?"
        ),
        "relevant_chunk_ids": [6],
    },
    {
        "query": (
            "How long after starting employment does medical "
            "coverage become active?"
        ),
        "relevant_chunk_ids": [7],
    },
]

def recall_at_k(retrieved_ids, relevant_ids):
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    if not relevant_set:
        return 0.0

    relevant_retrieved = retrieved_set & relevant_set

    return len(relevant_retrieved) / len(relevant_set)


def precision_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)

    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant_retrieved = [
        chunk_id
        for chunk_id in retrieved_ids[:k]
        if chunk_id in relevant_set
    ]

    return len(relevant_retrieved) / k


if __name__ == "__main__":
    k = 3

    total_recall = 0.0
    total_precision = 0.0

    for item in evaluation_dataset:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        # Retrieve the Top-K chunks
        results = retrieve(
            query,
            chunks,
            document_embeddings,
            model,
            top_k=k,
        )

        # Extract the retrieved Chunk IDs
        retrieved_ids = [
            chunk_id for chunk_id, score, chunk in results
        ]

        # Calculate metrics
        recall = recall_at_k(retrieved_ids, relevant_ids)
        precision = precision_at_k(retrieved_ids, relevant_ids, k)

        total_recall += recall
        total_precision += precision

        print(f"\nQuery: {query}")
        print(f"Relevant IDs: {relevant_ids}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Recall@{k}: {recall:.2%}")
        print(f"Precision@{k}: {precision:.2%}")

    # Calculate macro-average metrics
    num_queries = len(evaluation_dataset)

    avg_recall = total_recall / num_queries
    avg_precision = total_precision / num_queries

    print("\n=== Overall Evaluation ===")
    print(f"Number of queries: {num_queries}")
    print(f"Macro Recall@{k}: {avg_recall:.2%}")
    print(f"Macro Precision@{k}: {avg_precision:.2%}")