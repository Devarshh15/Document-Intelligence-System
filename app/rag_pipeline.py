# from app.ingest import load_documents
# from app.chunking import split_documents
# from app.embeddings import get_embedding_model
# from app.vector_store import create_vector_store
# from app.retriever import get_retriever
# from app.llm import get_llm_client, generate_answer


# # class RAGPipeline:

# #     def __init__(self):

# #         self.client = get_llm_client()

# #         self.vector_store = None
# #         self.retriever = None

# #         self.build_index()


# #     def build_index(self):

# #         documents = load_documents()

# #         if len(documents) == 0:
# #             return

# #         chunks = split_documents(documents)

# #         embeddings = get_embedding_model()

# #         self.vector_store = create_vector_store(chunks, embeddings)

# #         self.retriever = get_retriever(self.vector_store)


# #     def ask(self, query):

# #         if self.retriever is None:
# #             return "No documents uploaded yet.", []

# #         docs = self.retriever.invoke(query)

# #         context = "\n\n".join([doc.page_content for doc in docs])

# #         sources = []

# #         for doc in docs:
# #             source = doc.metadata.get("source", "")
# #             page = doc.metadata.get("page", "")
# #             sources.append(f"{source} page {page}")

# #         answer = generate_answer(self.client, query, context)

# #         if "not available" in answer.lower():
# #             sources = []

# #         return answer, sources




# from app.ingest import load_documents
# from app.chunking import split_documents
# from app.embeddings import get_embedding_model
# from app.vector_store import create_vector_store
# from app.retriever import get_retriever
# from app.llm import get_llm_client, generate_answer
# import datetime


# def log_query(query, answer, sources):

#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     with open("logs.txt", "a", encoding="utf-8") as f:
#         f.write(f"\n[{timestamp}]\n")
#         f.write(f"Query: {query}\n")
#         f.write(f"Answer: {answer}\n")
#         f.write(f"Sources: {sources}\n")
#         f.write("-" * 50 + "\n")


# class RAGPipeline:

#     def __init__(self):

#         self.client = get_llm_client()

#         self.vector_store = None
#         self.retriever = None

#         self.build_index()


#     def build_index(self):
#         """
#         Build or rebuild FAISS index from documents folder
#         """

#         documents = load_documents()

#         if len(documents) == 0:
#             self.vector_store = None
#             self.retriever = None
#             return

#         chunks = split_documents(documents)

#         embeddings = get_embedding_model()

#         self.vector_store = create_vector_store(chunks, embeddings)

#         self.retriever = get_retriever(self.vector_store)


#     def ask(self, query):
#         """
#         Process user query through RAG pipeline
#         """

#         if self.retriever is None:
#             return "No documents uploaded yet.", []

#         docs = self.retriever.invoke(query)

#         answer = generate_answer(self.client, query, docs)

#         # sources = []

#         # # Only show sources if answer actually comes from documents
#         # if "not available" not in answer.lower():

#         #     for doc in docs:

#         #         source = doc.metadata.get("source", "")

#         #         page = doc.metadata.get("page", "")

#         #         if page != "":
#         #             sources.append(f"{source} page {page}")
#         #         else:
#         #             sources.append(source)


#         # sources = []

#         # if "not available" not in answer.lower():

#         #     seen_sources = set()

#         #     for doc in docs:

#         #         source = doc.metadata.get("source", "")

#         #         page = doc.metadata.get("page", "")

#         #         if page != "":
#         #             source_entry = f"{source} page {page}"
#         #         else:
#         #             source_entry = source

#         #     # Deduplicate sources
#         #     if source_entry not in seen_sources:
#         #         seen_sources.add(source_entry)
#         #         sources.append(source_entry)




#         # sources = []

#         # if "not available" not in answer.lower():

#         #     seen_sources = set()

#         #     for doc in docs:

#         #         source = doc.metadata.get("source", "")

#         #          # clean file name
#         #         source = source.split("\\")[-1].split("/")[-1]

#         #         if source not in seen_sources:
#         #             seen_sources.add(source)
#         #             sources.append(source)


#         sources = []

#         if "not available" not in answer.lower():

#             seen_sources = set()

#              # reuse embedding model for similarity filtering
#             embedding_model = get_embedding_model()

#             answer_vector = embedding_model.embed_query(answer)

#             for doc in docs:

#                 source = doc.metadata.get("source", "")
#                 source = source.split("\\")[-1].split("/")[-1]

#                 chunk_vector = embedding_model.embed_query(doc.page_content)

#                 # cosine similarity
#                 similarity = sum(a*b for a, b in zip(answer_vector, chunk_vector))

#                 # threshold tuned for MiniLM embeddings
#                 if similarity > 0.55:

#                     if source not in seen_sources:
#                         seen_sources.add(source)
#                         sources.append(source)


#         log_query(query, answer, sources)

#         return answer, sources



from app.ingest import load_documents
from app.chunking import split_documents
from app.embeddings import get_embedding_model
from app.vector_store import create_vector_store
from app.retriever import get_retriever
from app.llm import get_llm_client, generate_answer
from app.hybrid_retriever import HybridRetriever
from app.reranker import CrossEncoderReranker

import datetime
import os


# ---------------- CONFIG ----------------
USE_HYBRID_RETRIEVAL = True
USE_RERANKER = True
RERANK_TOP_K = 3
# ----------------------------------------


def log_query(query, answer, sources):

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n")
        f.write(f"Query: {query}\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"Sources: {sources}\n")
        f.write("-" * 50 + "\n")


class RAGPipeline:

    def __init__(self, documents_folder="documents"):

        self.documents_folder = documents_folder
        self.client = get_llm_client()
        self.embedding_model = get_embedding_model()

        self.vector_store = None
        self.retriever = None
        self.documents = None
        self.load_errors = {}

        # NEW: reranker
        self.reranker = None

        self.build_index()


    def build_index(self):

        os.makedirs(self.documents_folder, exist_ok=True)

        documents, load_errors = load_documents(self.documents_folder)
        
        # Store errors for access in UI
        self.load_errors = load_errors

        if len(documents) == 0:
            self.vector_store = None
            self.retriever = None
            return

        self.documents = documents

        chunks = split_documents(documents)

        embeddings = self.embedding_model
        self.vector_store = create_vector_store(chunks, embeddings)

        base_retriever = get_retriever(self.vector_store)

        # HYBRID
        if USE_HYBRID_RETRIEVAL:
            self.retriever = HybridRetriever(base_retriever, chunks)
        else:
            self.retriever = base_retriever

        # RERANKER INIT
        if USE_RERANKER:
            self.reranker = CrossEncoderReranker(top_k=RERANK_TOP_K)

    

#     def ask(self, query):
#         """
#         Process user query through RAG pipeline
# """

#         if self.retriever is None:
#             return "No documents uploaded yet.", [], None

#         docs = self.retriever.invoke(query)

#         # RERANK
#         if USE_RERANKER and self.reranker:
#             docs = self.reranker.rerank(query, docs)

#         answer = generate_answer(self.client, query, docs)

#         sources = []
#         confidence = 0.0

#         embedding_model = get_embedding_model()

#         if "not available" not in answer.lower():

#             seen_sources = set()

#             answer_vector = embedding_model.embed_query(answer)

#             similarities = []

#             for doc in docs:

#                 source = doc.metadata.get("source", "")
#                 source = source.split("\\")[-1].split("/")[-1]

#                 chunk_vector = embedding_model.embed_query(doc.page_content)

#                 similarity = sum(a*b for a, b in zip(answer_vector, chunk_vector))
#                 similarities.append(similarity)

#             if similarity > 0.55:
#                 if source not in seen_sources:
#                     seen_sources.add(source)
#                     sources.append(source)

#         # compute confidence
#         if similarities:
#             confidence = sum(similarities) / len(similarities)

#         log_query(query, answer, sources)

#         return answer, sources, round(confidence, 2)


    def ask(self, query):

        if self.retriever is None:
            return "No documents uploaded yet.", []

        docs = self.retriever.invoke(query)

        # ---------------- RERANK STEP ----------------
        if USE_RERANKER and self.reranker:
            docs = self.reranker.rerank(query, docs)
        # ------------------------------------------------

        answer = generate_answer(self.client, query, docs)

        sources = []

        if answer.strip().rstrip(".").casefold() != "the information is not available in the documents":

            seen_sources = set()

            for doc in docs:

                source = str(doc.metadata.get("source") or "")
                source = source.split("\\")[-1].split("/")[-1]

                if source and source not in seen_sources:
                    seen_sources.add(source)
                    sources.append(source)

        log_query(query, answer, sources)

        return answer, sources