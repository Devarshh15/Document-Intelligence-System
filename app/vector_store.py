from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):
    """
    Create FAISS vector database from document chunks
    """

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store
