def build_context(reranked_results, max_chunks=3):
    """
    Build an LLM-ready context from reranked chunks.

    Args:
        reranked_results:
            Candidates in the format
            (chunk_id, cross_encoder_score, chunk_text).

        max_chunks:
            Maximum number of chunks to include.

    Returns:
        A formatted context string.
    """

    selected_results = reranked_results[:max_chunks]

    context_parts = []

    for chunk_id, score, chunk in selected_results:
        context_part = (
            f"[Chunk ID: {chunk_id}]\n"
            f"{chunk}"
        )

        context_parts.append(context_part)

    return "\n\n".join(context_parts)