from rag_pipeline import answer_question


query = "What is the company's maternity leave policy?"

answer = answer_question(query)

print("Question:")
print(query)

print("\nAnswer:")
print(answer)