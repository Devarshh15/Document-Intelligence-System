def get_retriever(vector_store):
    """
    Create retriever from the vector database
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    return retriever