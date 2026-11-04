import re

from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    Combines BM25 (keyword) + FAISS (vector) retrieval
    """

    def __init__(self, vector_retriever, documents, k=5):

        self.vector_retriever = vector_retriever
        self.documents = documents
        self.k = k

        # Prepare BM25
        self.texts = [doc.page_content for doc in documents]
        self.tokenized_corpus = [re.findall(r"\w+", text.casefold()) for text in self.texts]

        self.bm25 = BM25Okapi(self.tokenized_corpus)


    def bm25_search(self, query):

        tokenized_query = re.findall(r"\w+", query.casefold())

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = [self.documents[i] for i in ranked_indices[:self.k]]

        return results


    def invoke(self, query):
        """
        Return unique BM25 and FAISS candidates for downstream reranking.
        """

        # FAISS results
        vector_docs = self.vector_retriever.invoke(query)

        # BM25 results
        bm25_docs = self.bm25_search(query)

        # Merge + deduplicate
        combined = []

        seen = set()

        for doc in vector_docs + bm25_docs:

            content = doc.page_content

            if content not in seen:
                seen.add(content)
                combined.append(doc)

        return combined