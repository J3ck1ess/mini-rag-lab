from prompt_builder import build_prompt
from llm_client import generate


query = "What is the company's maternity leave policy?"

context = """
[Chunk ID: 4]
Chapter 2: Remote Work

Remote work requires approval from the direct supervisor.

Employees must submit remote work requests through the company portal.
"""

prompt = build_prompt(query, context)

answer = generate(prompt)

print("Question:")
print(query)

print("\nContext:")
print(context)

print("\nGenerated Answer:")
print(answer)