from app.rag_pipeline import RAGPipeline
from app.evaluation import evaluate_rag

rag = RAGPipeline()

query = "What is leave policy?"

answer, sources = rag.ask(query)

# retrieve raw docs again for evaluation
docs = rag.retriever.invoke(query)
contexts = [doc.page_content for doc in docs]

scores = evaluate_rag(query, answer, contexts)

print("\nQuery:", query)
print("\nAnswer:", answer)
print("\nSources:", sources)

print("\n--- RAGAS SCORES ---")
print(scores)