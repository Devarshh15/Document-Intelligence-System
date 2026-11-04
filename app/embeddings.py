from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load embedding model used to convert text chunks into vectors
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings