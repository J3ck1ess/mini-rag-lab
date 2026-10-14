from llm_client import generate


prompt = """
Explain Retrieval-Augmented Generation in two sentences.
Use simple English.
"""

answer = generate(prompt)

print("Generated Answer:")
print(answer)