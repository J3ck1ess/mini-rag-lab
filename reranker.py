def rerank(query, candidates, reranker):
    """
    Rerank retrieved candidates using a Cross-Encoder.

    Args:
        query: The user's query.
        candidates: Retrieved chunks in the format
                    (chunk_id, embedding_score, chunk_text).
        reranker: A loaded CrossEncoder model.

    Returns:
        Candidates sorted by Cross-Encoder score in descending order.
        Each result has the format:
        (chunk_id, cross_encoder_score, chunk_text).
    """

    # 1. Prepare query-document pairs
    pairs = [
        (query, chunk)
        for _, _, chunk in candidates
    ]

    # 2. Score each pair
    scores = reranker.predict(pairs)

    # 3. Attach scores to the original candidates
    reranked_candidates = [
        (chunk_id, float(score), chunk)
        for (chunk_id, _, chunk), score
        in zip(candidates, scores)
    ]

    # 4. Sort by Cross-Encoder score
    reranked_candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return reranked_candidates