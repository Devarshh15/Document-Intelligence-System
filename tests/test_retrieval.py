import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.chunking import split_documents
from app.hybrid_retriever import HybridRetriever
from app.rag_pipeline import RAGPipeline
from app.retriever import get_retriever


class RetrievalTests(unittest.TestCase):
    def test_related_details_stay_in_one_chunk(self):
        text = "Technical overview. " * 40 + "Storage details belong to this section."
        document = Document(page_content=text, metadata={"source": "example.pdf"})
        chunks = split_documents([document])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, text)
        self.assertEqual(chunks[0].metadata, document.metadata)

    def test_vector_search_fetches_eight_candidates(self):
        store = Mock()
        self.assertIs(get_retriever(store), store.as_retriever.return_value)
        store.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 8}
        )

    def test_keyword_search_ignores_case_and_punctuation(self):
        documents = [
            Document(page_content="The ARCHIVE, stores retained records."),
            Document(page_content="Tickets are assigned to agents."),
            Document(page_content="Reports refresh each morning."),
        ]
        retriever = HybridRetriever(Mock(), documents, k=1)
        self.assertIs(retriever.bm25_search("archive?")[0], documents[0])

    def test_shared_headers_do_not_hide_distinct_chunks(self):
        header = "Repeated document header " * 6
        first = Document(page_content=header + "First section.")
        second = Document(page_content=header + "Second section.")
        vector = Mock()
        vector.invoke.return_value = [first, second]
        retriever = HybridRetriever(vector, [first, second])
        retriever.bm25_search = Mock(return_value=[first, second])
        self.assertEqual(retriever.invoke("section"), [first, second])

    def test_candidates_are_not_truncated_before_reranking(self):
        documents = [Document(page_content=f"Section {index}") for index in range(8)]
        vector = Mock()
        vector.invoke.return_value = documents[:5]
        retriever = HybridRetriever(vector, documents)
        retriever.bm25_search = Mock(return_value=documents[3:])
        self.assertEqual(retriever.invoke("section"), documents)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = RAGPipeline.__new__(RAGPipeline)
        self.pipeline.client = Mock()
        self.pipeline.retriever = Mock()
        self.pipeline.reranker = Mock()
        self.pipeline.embedding_model = Mock()

    def ask_with(self, answer, documents):
        self.pipeline.retriever.invoke.return_value = documents
        self.pipeline.reranker.rerank.return_value = documents
        with patch("app.rag_pipeline.generate_answer", return_value=answer) as generate:
            with patch("app.rag_pipeline.log_query"):
                result = self.pipeline.ask("A short question?")
            generate.assert_called_once_with(
                self.pipeline.client, "A short question?", documents
            )
        self.pipeline.embedding_model.embed_query.assert_not_called()
        return result

    def test_short_answer_keeps_source_of_long_context(self):
        document = Document(
            page_content="Section detail. " * 100,
            metadata={"source": "report.pdf"},
        )
        self.assertEqual(
            self.ask_with("A brief answer.", [document]),
            ("A brief answer.", ["report.pdf"]),
        )

    def test_sources_only_come_from_final_reranked_context(self):
        discarded = Document(page_content="Unselected", metadata={"source": "other.pdf"})
        selected = Document(page_content="Selected", metadata={"source": "chosen.pdf"})
        self.pipeline.retriever.invoke.return_value = [discarded, selected]
        self.pipeline.reranker.rerank.return_value = [selected]
        with patch("app.rag_pipeline.generate_answer", return_value="An answer.") as generate:
            with patch("app.rag_pipeline.log_query"):
                self.assertEqual(self.pipeline.ask("Question?"), ("An answer.", ["chosen.pdf"]))
            generate.assert_called_once_with(self.pipeline.client, "Question?", [selected])

    def test_refusal_does_not_show_retrieved_sources(self):
        document = Document(page_content="Unrelated context", metadata={"source": "report.pdf"})
        for answer in [
            "The information is not available in the documents.",
            "  THE INFORMATION IS NOT AVAILABLE IN THE DOCUMENTS  ",
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(self.ask_with(answer, [document]), (answer, []))

    def test_supported_negative_answer_keeps_sources(self):
        answer = "Live chat is not available in the first release."
        document = Document(page_content=answer, metadata={"source": "scope.pdf"})
        self.assertEqual(self.ask_with(answer, [document]), (answer, ["scope.pdf"]))

    def test_paths_duplicates_and_missing_metadata(self):
        documents = [
            Document(page_content="First", metadata={"source": r"C:\docs\report.pdf"}),
            Document(page_content="Second", metadata={"source": "/docs/report.pdf"}),
            Document(page_content="Third", metadata={"source": "/docs/appendix.pdf"}),
            Document(page_content="Missing"),
            Document(page_content="Empty", metadata={"source": None}),
        ]
        self.assertEqual(
            self.ask_with("Combined answer.", documents),
            ("Combined answer.", ["report.pdf", "appendix.pdf"]),
        )

    def test_no_documents_does_not_generate_answer(self):
        self.pipeline.retriever = None
        with patch("app.rag_pipeline.generate_answer") as generate:
            self.assertEqual(self.pipeline.ask("Question?"), ("No documents uploaded yet.", []))
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()