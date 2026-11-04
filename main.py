from app.ingest import load_documents
from app.chunking import split_documents
from app.embeddings import get_embedding_model
from app.vector_store import create_vector_store
from app.retriever import get_retriever
from app.llm import get_llm_client, generate_answer


# Step 1: Load documents
documents = load_documents()

# Step 2: Chunk documents
chunks = split_documents(documents)

# Step 3: Load embeddings
embeddings = get_embedding_model()

# Step 4: Create vector database
vector_store = create_vector_store(chunks, embeddings)

# Step 5: Create retriever
retriever = get_retriever(vector_store)

# Step 6: Initialize Grok
client = get_llm_client()


print("\nAskMyDocs is ready\n")


while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Exiting AskMyDocs assistant.")
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    # answer = generate_answer(client, query, context)

    result = generate_answer(client, query, docs)

    print("\nAnswer:\n")
    print(result["answer"])
    print("\nSources:\n")
    for source in result["sources"]:
        print(f" - {source}")
    print("\n---------------------------------\n")


# while True:

#     query = input("Ask a question: ")

#     # retrieve relevant chunks
#     docs = retriever.invoke(query)

#     context = "\n\n".join([doc.page_content for doc in docs])

#     # generate answer
#     answer = generate_answer(client, query, context)

#     print("\nAnswer:\n")
#     print(answer)
#     print("\n---------------------------------\n")