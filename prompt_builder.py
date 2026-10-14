def build_prompt(query, context):
    prompt = f"""
You are a helpful assistant that answers questions using the provided context.

Rules:
1. Answer using only the information in the context.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain the answer, say:
   "I cannot answer this question based on the provided context."

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt.strip()