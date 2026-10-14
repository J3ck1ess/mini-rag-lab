from sentence_transformers import CrossEncoder

from mini_rag import (
    chunks,
    document_embeddings,
    model,
    retrieve,
)
from evaluation import evaluation_dataset
from reranker import rerank


def reciprocal_rank(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids[:k])

    if not relevant_set:
        return 0.0

    return len(retrieved_set & relevant_set) / len(relevant_set)


def recall_at_1(retrieved_ids, relevant_ids):
    if not retrieved_ids:
        return 0.0

    return float(retrieved_ids[0] in set(relevant_ids))


def evaluate_reranking(top_k=3):
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    baseline_recalls = []
    reranked_recalls = []

    baseline_mrrs = []
    reranked_mrrs = []

    baseline_recall_at_3 = []
    baseline_recall_at_5 = []

    reranked_recall_at_3 = []
    reranked_recall_at_5 = []

    for item in evaluation_dataset:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        # Step 1: Retrieve candidates using the embedding model
        candidates = retrieve(
            query=query,
            chunks=chunks,
            document_embeddings=document_embeddings,
            model=model,
            top_k=top_k,
        )

        baseline_ids = [
            chunk_id
            for chunk_id, _, _ in candidates
        ]

        # Step 2: Rerank candidates using Cross-Encoder
        reranked_candidates = rerank(
            query=query,
            candidates=candidates,
            reranker=reranker,
        )

        reranked_ids = [
            chunk_id
            for chunk_id, _, _ in reranked_candidates
        ]

        # Step 3: Calculate metrics
        baseline_recalls.append(
            recall_at_1(baseline_ids, relevant_ids)
        )

        reranked_recalls.append(
            recall_at_1(reranked_ids, relevant_ids)
        )

        baseline_mrrs.append(
            reciprocal_rank(baseline_ids, relevant_ids, k=3)
        )

        reranked_mrrs.append(
            reciprocal_rank(reranked_ids, relevant_ids, k=3)
        )

        baseline_recall_at_3.append(
            recall_at_k(baseline_ids, relevant_ids, k=3)
        )

        baseline_recall_at_5.append(
            recall_at_k(baseline_ids, relevant_ids, k=5)
        )

        reranked_recall_at_3.append(
            recall_at_k(reranked_ids, relevant_ids, k=3)
        )

        reranked_recall_at_5.append(
            recall_at_k(reranked_ids, relevant_ids, k=5)
        )

        # Step 4: Print per-query results
        print(f"\nQuery: {query}")
        print(f"Relevant Chunk IDs: {relevant_ids}")
        print(f"Baseline Top-{len(baseline_ids)}: {baseline_ids}")
        print(f"Reranked Top-{len(reranked_ids)}: {reranked_ids}")
        print(
            f"Recall@1: "
            f"{baseline_recalls[-1]:.3f} -> "
            f"{reranked_recalls[-1]:.3f}"
        )
        print(
            f"MRR@3: "
            f"{baseline_mrrs[-1]:.3f} -> "
            f"{reranked_mrrs[-1]:.3f}"
        )

    # Step 5: Calculate macro averages
    n = len(evaluation_dataset)

    baseline_recall = sum(baseline_recalls) / n
    reranked_recall = sum(reranked_recalls) / n

    baseline_mrr = sum(baseline_mrrs) / n
    reranked_mrr = sum(reranked_mrrs) / n

    print("\n" + "=" * 55)
    print("FINAL EVALUATION")
    print("=" * 55)

    print(
        f"Recall@1: "
        f"{baseline_recall:.2%} -> {reranked_recall:.2%}"
    )

    print(
        f"MRR@3: "
        f"{baseline_mrr:.2%} -> {reranked_mrr:.2%}"
    )

    print(
        f"Recall@3: "
        f"{sum(baseline_recall_at_3) / n:.2%} -> "
        f"{sum(reranked_recall_at_3) / n:.2%}"
    )

    print(
        f"Recall@5: "
        f"{sum(baseline_recall_at_5) / n:.2%} -> "
        f"{sum(reranked_recall_at_5) / n:.2%}"
    )


if __name__ == "__main__":
    evaluate_reranking(top_k=5)