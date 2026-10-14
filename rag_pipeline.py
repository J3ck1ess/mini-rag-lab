from sentence_transformers import CrossEncoder

from mini_rag import chunks, document_embeddings, model, retrieve
from reranker import rerank
from context_builder import build_context
from prompt_builder import build_prompt
from llm_client import generate

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def answer_question(query):
    # Step 1: Retrieve candidate chunks
    candidates = retrieve(
        query,
        chunks,
        document_embeddings,
        model,
        top_k=5,
    )

    # Step 2: Rerank candidates
    reranked_results = rerank(
        query,
        candidates,
        reranker_model,
    )

    # Step 3: Build context
    context = build_context(
        reranked_results,
        max_chunks=3,
    )

    # Step 4: Build prompt
    prompt = build_prompt(query, context)

    # Step 5: Generate answer
    answer = generate(prompt)

    return answer