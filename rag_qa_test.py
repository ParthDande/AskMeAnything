from transformers import pipeline

# Load a simple QA model (small and works without GPU too)
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Just a test to check
question = "What is RAG?"
context = "RAG stands for Retrieval Augmented Generation. It helps LLMs answer questions based on external documents."

result = qa(question=question, context=context)
print("Answer:", result['answer'])
