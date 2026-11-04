from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Re-ranks retrieved documents using a cross-encoder model
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=3):

        self.model = CrossEncoder(model_name)
        self.top_k = top_k


    def rerank(self, query, documents):
        """
        Rank documents based on relevance to query
        """

        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k documents
        reranked_docs = [doc for doc, score in scored_docs[:self.top_k]]

        return reranked_docs