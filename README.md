# AskMyDocs

AskMyDocs is a document-grounded AI assistant that answers questions from uploaded PDF, TXT, Word, and PowerPoint files. It uses a Retrieval-Augmented Generation (RAG) pipeline so answers are based on the contents of the documents in the `documents/` folder instead of general model knowledge.

The project currently provides three ways to use the app:

- **Streamlit web app**: upload documents and ask questions from a browser UI.
- **Command-line assistant**: ask questions from a terminal.
- **FastAPI service**: expose document upload and question-answering endpoints, including OpenAI-compatible chat routes for tools such as Open WebUI.

## What The App Does

AskMyDocs loads local documents, splits them into searchable chunks, embeds those chunks into vectors, retrieves the most relevant content for a question, and sends only that retrieved context to a Groq-hosted LLM. The model is instructed to answer strictly from the retrieved document context. If the requested information is not found in the documents, the app should respond with:

```text
The information is not available in the documents.
```

The app also returns source file names when the answer is supported by matching retrieved chunks.

## Main Features

- **Multi-format document support**: Upload and index PDF, TXT, Word (.docx, .doc), and PowerPoint (.pptx, .ppt) files.
- **Natural language Q&A**: Ask questions in plain English about your documents.
- **LLM-powered answers**: Generate accurate, context-grounded responses using Groq's language models (primary: `openai/gpt-oss-20b`, fallback: `openai/gpt-oss-120b`).
- **Local embeddings**: Use Hugging Face sentence-transformer embeddings (`all-MiniLM-L6-v2`) to avoid sending document content to external services.
- **Vector database**: Store and search embeddings using an in-memory FAISS vector database for fast retrieval.
- **Hybrid retrieval**: Combine semantic search (vector similarity) with BM25 keyword retrieval for better recall.
- **Smart re-ranking**: Re-rank retrieved chunks using a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) to ensure the most relevant context is sent to the LLM.
- **Source tracking**: Return source document filenames with answers for provenance and verification.
- **Query logging**: Maintain a complete log of all queries, answers, and sources in `logs.txt` for auditing.
- **Three deployment modes**: Run as a Streamlit web UI for ease of use, a command-line interface for automation, or a FastAPI service for programmatic access and integration with tools like Open WebUI.

## Project Structure

```text
AskMyDocs/
├── api.py                     # FastAPI application and API endpoints
├── app_ui.py                  # Streamlit browser UI
├── main.py                    # Command-line interface
├── requirements.txt           # Python dependency list
├── logs.txt                   # Query/answer/source log file
├── documents/                 # Uploaded or manually added supported files
└── app/
    ├── chunking.py            # Splits loaded documents into chunks
    ├── embeddings.py          # Loads Hugging Face embedding model
    ├── evaluation.py          # Optional RAGAS evaluation helper
    ├── evaluate_example.py    # Example evaluation script
    ├── hybrid_retriever.py    # Combines BM25 and vector retrieval
    ├── ingest.py              # Loads PDF and TXT documents
    ├── llm.py                 # Groq client and prompt-based answer generation
    ├── rag_pipeline.py        # End-to-end RAG orchestration
    ├── reranker.py            # Cross-encoder re-ranking
    ├── retriever.py           # FAISS retriever configuration
    └── vector_store.py        # FAISS vector store creation
```

## How The RAG Pipeline Works

1. **Document loading**
   - `app/ingest.py` reads all supported files from the `documents/` folder.
  - PDF files are loaded with `PyPDFLoader`.
  - TXT files are loaded with `TextLoader`.
  - DOCX files are loaded with `Docx2txtLoader`.
  - DOC, PPT, and PPTX files are loaded with Unstructured loaders. Legacy `.doc` and `.ppt` files may require LibreOffice to be installed on the machine running the app.

2. **Chunking**
   - `app/chunking.py` uses `RecursiveCharacterTextSplitter`.
  - Each chunk is up to `1500` characters to keep related section details together.
  - Chunks overlap by `200` characters so nearby context is not lost between chunks.

3. **Embedding**
   - `app/embeddings.py` loads `sentence-transformers/all-MiniLM-L6-v2` through Hugging Face embeddings.
   - Each document chunk is converted into a numeric vector.

4. **Vector storage**
   - `app/vector_store.py` creates a FAISS vector database from the chunks.
   - The current implementation builds the FAISS index in memory each time the pipeline builds or rebuilds the index.

5. **Retrieval**
  - `app/retriever.py` creates a similarity retriever with `k=8` candidates.
   - `app/hybrid_retriever.py` can combine FAISS semantic retrieval with BM25 keyword retrieval.
  - Keyword search normalizes case and punctuation and retrieves up to `5` candidates.
  - The hybrid retriever passes all unique candidates to reranking, comparing full chunk text instead of shared headers.

6. **Re-ranking**
   - `app/reranker.py` uses `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - The re-ranker scores retrieved chunks against the user question.
   - The top `3` chunks are passed to answer generation.

7. **Answer generation**
   - `app/llm.py` sends the question and retrieved context to Groq.
   - The prompt instructs the model to use only the provided context and respond with "The information is not available in the documents." if the answer is not found.
   - Primary Groq model: `openai/gpt-oss-20b`.
   - Fallback model: `openai/gpt-oss-120b` (used if the primary model is unavailable).
   - If both models fail, the app raises an error with details from all attempts.

8. **Source provenance and logging**
  - `app/rag_pipeline.py` lists unique filenames from the final reranked chunks supplied to answer generation.
  - Sources identify the documents provided as context, not independently verified evidence for each claim.
  - Document-not-found refusals have no sources. Answer length and chunk length do not suppress source labels.
   - Each query, answer, and source list is appended to `logs.txt`.

## Prerequisites

Install these before starting:

- Python 3.10 or newer recommended.
- A Groq API key.
- Internet access for the first run, because Hugging Face models are downloaded automatically.

You can check your Python version with:

```powershell
python --version
```

If `python` is not recognized on Windows, try:

```powershell
py --version
```

## Setup On Windows PowerShell

Run these commands from PowerShell.

### 1. Open The Project Folder

```powershell
cd "c:\Users\4000036\Downloads\AskMyDocs"
```

### 2. Create A Virtual Environment

```powershell
python -m venv .venv
```

If your system uses the Python launcher instead, run:

```powershell
py -m venv .venv
```

### 3. Activate The Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation with an execution policy error, run this once for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

After activation, your prompt should show `(.venv)`.

### 4. Upgrade Pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install Dependencies

Install the dependencies listed in the project:

```powershell
pip install -r requirements.txt
```

### 6. Create The Environment File

Create a file named `.env` in the project root:

```powershell
New-Item -ItemType File -Path .env
```

Open `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace `your_groq_api_key_here` with your real Groq API key.

### 7. Make Sure The Documents Folder Exists

```powershell
New-Item -ItemType Directory -Force -Path documents
```

You can add files manually by copying supported files into the folder:

```powershell
Copy-Item "C:\path\to\your\file.pdf" -Destination ".\documents\"
```

The Streamlit UI and API can also upload files for you.

## Setup On macOS Or Linux

Use these commands if you are running the project outside Windows:

```bash
cd /path/to/AskMyDocs
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p documents
```

Create `.env` in the project root:

```bash
touch .env
```

Add this line to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Start The Streamlit Web App

The Streamlit app is the easiest way to use AskMyDocs.

From the project root, with the virtual environment activated, run:

```powershell
streamlit run app_ui.py
```

Streamlit usually starts at:

```text
http://localhost:8501
```

## Deploy On Streamlit Community Cloud

For Streamlit Community Cloud, deploy `app_ui.py` as the app entrypoint.

Before deploying, add these secrets in **App Settings -> Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_FALLBACK_MODELS = "openai/gpt-oss-120b"
```

The repository includes:

- `runtime.txt` to pin Python 3.11.
- `packages.txt` to install Linux packages needed by legacy Office document loaders.
- `.streamlit/config.toml` for Streamlit runtime settings.

Uploaded files are stored in a temporary per-session folder on Streamlit Cloud, then indexed for that user's current session. Files are not permanently stored by the deployed app.

Open that URL in your browser.

### Streamlit Usage Steps

1. Start the app with `streamlit run app_ui.py`.
2. Open `http://localhost:8501`.
3. Upload one or more PDF or TXT files.
4. Wait for the success message after upload.
5. Type a question in the question box.
6. Click **Ask**.
7. Review the answer and the listed sources.
8. Use **Clear Question Cache** if you want to reset the current question and cached answer state.

When files are uploaded through the UI, they are saved into the `documents/` folder and the RAG index is rebuilt.

## Start The Command-Line App

The CLI version reads documents from the `documents/` folder and lets you ask questions in the terminal.

Before starting, place at least one PDF or TXT file in `documents/`:

```powershell
Copy-Item "C:\path\to\your\file.pdf" -Destination ".\documents\"
```

Then run:

```powershell
python main.py
```

You should see:

```text
AskMyDocs is ready
```

Then ask a question:

```text
Ask a question (type 'exit' to quit): What is this document about?
```

To stop the CLI app, type:

```text
exit
```

## Start The FastAPI App

The API version exposes endpoints for health checks, file uploads, direct question answering, and OpenAI-compatible chat completion.

From the project root, with the virtual environment activated, run:

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### API Health Check

PowerShell command:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/" -Method Get
```

Expected response shape:

```json
{
  "name": "AskMyDocs",
  "tagline": "AI that answers from your documents, not assumptions.",
  "message": "AskMyDocs API is running"
}
```

### Upload Documents Through The API

PowerShell command:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload-documents" -F "files=@C:\path\to\your\file.pdf"
```

To upload more than one file:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload-documents" -F "files=@C:\path\to\file1.pdf" -F "files=@C:\path\to\file2.txt"
```

After upload, the API saves the files into `documents/` and rebuilds the RAG index.

### Ask A Question Through The API

PowerShell command:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is this document about?"}'
```

Expected response shape:

```json
{
  "answer": "...",
  "sources": ["example.pdf"]
}
```

If the answer cannot be found, the response may be:

```json
{
  "answer": "The information is not available in the documents."
}
```

### List OpenAI-Compatible Models

PowerShell command:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/models" -Method Get
```

Expected response shape:

```json
{
  "data": [
    {
      "id": "askmydocs-model",
      "object": "model"
    }
  ]
}
```

### Use The OpenAI-Compatible Chat Endpoint

PowerShell command:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"model":"askmydocs-model","messages":[{"role":"user","content":"What is this document about?"}]}'
```

Expected response shape:

```json
{
  "id": "askmydocs-response",
  "object": "chat.completion",
  "model": "askmydocs-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

## Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Yes | None | Groq API key. Used by `app/llm.py` to authenticate with Groq API and generate answers. |
| `GROQ_MODEL` | No | `openai/gpt-oss-20b` | Primary Groq model ID. Tried first for all requests. |
| `GROQ_FALLBACK_MODELS` | No | `openai/gpt-oss-120b` | Comma-separated list of fallback Groq model IDs. Tried in order if the primary model fails or is unavailable. |

The project loads environment variables using `python-dotenv`. For local development, create a `.env` file in the project root and add your configuration values there. For Streamlit Cloud, add variables to **App Settings -> Secrets**.

## Supported Document Types

| File Type | Supported | Loader |
| --- | --- | --- |
| `.pdf` | Yes | `PyPDFLoader` |
| `.txt` | Yes | `TextLoader` |
| `.docx` | Yes | `Docx2txtLoader` |
| `.doc` | Yes | `UnstructuredWordDocumentLoader` |
| `.pptx` | Yes | `UnstructuredPowerPointLoader` |
| `.ppt` | Yes | `UnstructuredPowerPointLoader` |

Other file types are ignored by the current loader. Legacy `.doc` and `.ppt` files may require LibreOffice at runtime.

## Logs

Each question is logged to `logs.txt` with:

- Timestamp.
- User query.
- Generated answer.
- Source list.

Example log shape:

```text
[2026-08-06 12:00:00]
Query: What is this document about?
Answer: ...
Sources: ['example.pdf']
--------------------------------------------------
```

## Troubleshooting

### `GROQ_API_KEY is not set`

Cause: `.env` is missing or does not contain `GROQ_API_KEY`.

Fix:

```powershell
New-Item -ItemType File -Force -Path .env
```

Then add:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_FALLBACK_MODELS=openai/gpt-oss-120b
```

Restart the app after editing `.env`.

### `ModuleNotFoundError: No module named 'fastapi'`

Cause: API dependencies are not installed.

Fix:

```powershell
pip install fastapi uvicorn python-multipart
```

### `ModuleNotFoundError: No module named 'langchain_huggingface'`

Cause: the Hugging Face LangChain integration package is not installed.

Fix:

```powershell
pip install langchain-huggingface
```

### `ModuleNotFoundError: No module named 'langchain_text_splitters'`

Cause: the LangChain text splitters package is not installed.

Fix:

```powershell
pip install langchain-text-splitters
```

### `ModuleNotFoundError: No module named 'rank_bm25'`

Cause: the BM25 package used by the hybrid retriever is not installed.

Fix:

```powershell
pip install rank-bm25
```

### The First Run Is Slow

Cause: the embedding model and cross-encoder model may download from Hugging Face on first use.

Fix: wait for the download to complete. Later runs should be faster because the models are cached locally.

### `No documents uploaded yet.`

Cause: the `documents/` folder is empty, or the app has not rebuilt the index after files were added.

Fix:

- For Streamlit, upload files through the UI.
- For API, call `/upload-documents`.
- For CLI, copy PDF or TXT files into `documents/` before running `python main.py`.

### Answers Do Not Include Sources

Cause: no documents are indexed, the answer is the document-not-found refusal, or the final context chunks have no source filename metadata.

Fix: upload and index documents containing the answer. Custom document loaders must preserve the `source` metadata field. Source filenames are taken only from the context supplied to the model, never invented from the answer text.

## Common Development Commands

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the Streamlit UI:

```powershell
streamlit run app_ui.py
```

Run the CLI:

```powershell
python main.py
```

Run the API:

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Install all known runtime packages:

```powershell
pip install -r requirements.txt
```

Deactivate the virtual environment:

```powershell
deactivate
```

## Recommended Startup Path

For most users, use this exact sequence:

```powershell
cd "c:\Users\4000036\Downloads\AskMyDocs"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
New-Item -ItemType File -Force -Path .env
New-Item -ItemType Directory -Force -Path documents
```

Add your Groq key to `.env` before starting the app:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Then start the Streamlit app:

```powershell
streamlit run app_ui.py
```