# from fastapi import FastAPI, UploadFile, File
# from pydantic import BaseModel
# from app.rag_pipeline import RAGPipeline
# import shutil
# import os

# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi import Request


# # app = FastAPI()


# from fastapi import FastAPI

# app = FastAPI()

# rag = RAGPipeline()  # initialize once (IMPORTANT for performance)

# @app.get("/")
# def home():
#     return {"message": "RAG API is running 🚀"}

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/ask")
# def ask_question(request: QueryRequest):
#     answer = rag.ask(request.query)
#     return {"answer": answer}


# @app.post("/ask")
# def ask_question(query: str):
#     answer = rag.run(query)   # or rag.query(query) based on your code
#     return {"answer": answer}


# # @app.post("/upload-documents")
# # async def upload_documents(files: list[UploadFile] = File(...)):

# #     saved_files = []

# #     for file in files:

# #         file_path = f"documents/{file.filename}"

# #         with open(file_path, "wb") as buffer:
# #             shutil.copyfileobj(file.file, buffer)

# #         saved_files.append(file.filename)

# #     return {
# #         "message": "Documents uploaded successfully",
# #         "files": saved_files
# #     }

# @app.post("/upload-documents")
# async def upload_documents(files: list[UploadFile] = File(...)):

#     saved_files = []

#     for file in files:

#         file_path = f"documents/{file.filename}"

#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         saved_files.append(file.filename)

#     # rebuild vector index
#     rag.build_index()

#     return {
#         "message": "Documents uploaded and indexed successfully",
#         "files": saved_files
#     }


# @app.post("/ask")
# async def ask_question(request: dict):

#     question = request["question"]

#     # answer, sources = rag.ask(question)

#     answer, sources = rag.ask(question)

#     # response = {
#     #     "answer": answer
#     # }

#     response = {
#     "answer": answer,
#     }

#     if sources:
#         response["sources"] = sources

#     return response




from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.ingest import SUPPORTED_EXTENSIONS
from app.rag_pipeline import RAGPipeline
import shutil
import os
from typing import List

app = FastAPI(
    title="AskMyDocs API",
    description="AI that answers from your documents, not assumptions.",
    version="1.0.0"
)

# ✅ Initialize RAG only once (important for performance)
rag = RAGPipeline()


# from fastapi import Request

# @app.post("/v1/chat/completions")
# async def chat_completions(request: Request):
#     body = await request.json()

#     user_message = body["messages"][-1]["content"]

#     # Run RAG
#     answer, sources = rag.ask(user_message)

#     return {
#         "id": "rag-response",
#         "object": "chat.completion",
#         "choices": [
#             {
#                 "message": {
#                     "role": "assistant",
#                     "content": answer
#                 }
#             }
#         ]
#     }



from fastapi import Request


@app.get("/v1/models")
def list_models():
    return {
        "data": [
            {
                "id": "askmydocs-model",
                "object": "model"
            }
        ]
    }



@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()

        # Validate input
        if "messages" not in body or len(body["messages"]) == 0:
            return {"error": "Invalid request: 'messages' missing"}

        user_message = body["messages"][-1]["content"]

        # Run RAG
        answer, sources = rag.ask(user_message)

        return {
            "id": "askmydocs-response",
            "object": "chat.completion",
            "model": "askmydocs-model",   # important for OpenWebUI
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    except Exception as e:
        return {
            "error": str(e)
        }




# ==============================
# 📌 Health Check
# ==============================
@app.get("/")
def home():
    return {
        "name": "AskMyDocs",
        "tagline": "AI that answers from your documents, not assumptions.",
        "message": "AskMyDocs API is running"
    }


# ==============================
# 📌 Request Schema
# ==============================
class QueryRequest(BaseModel):
    query: str


# ==============================
# 📌 Ask Question Endpoint
# ==============================
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        # Call your RAG pipeline
        answer, sources = rag.ask(request.query)

        response = {
            "answer": answer
        }

        # Include sources if available
        if sources:
            response["sources"] = sources

        return response

    except Exception as e:
        return {
            "error": str(e)
        }


# ==============================
# 📌 Upload Documents Endpoint
# ==============================
@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        # Ensure folder exists
        os.makedirs("documents", exist_ok=True)

        saved_files = []

        for file in files:
            extension = os.path.splitext(file.filename or "")[1].lower()

            if extension not in SUPPORTED_EXTENSIONS:
                supported_types = ", ".join(SUPPORTED_EXTENSIONS)
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type for {file.filename}. Supported types: {supported_types}"
                )

            file_path = f"documents/{file.filename}"

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append(file.filename)

        # Rebuild vector index after upload
        rag.build_index()

        return {
            "message": "Documents uploaded and indexed successfully",
            "files": saved_files
        }

    except HTTPException:
        raise

    except Exception as e:
        return {
            "error": str(e)
        }