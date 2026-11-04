import os
import zipfile
from typing import Tuple

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".doc", ".docx", ".ppt", ".pptx")


def is_valid_pptx(file_path: str) -> bool:
    """
    Validate if a PPTX file is not corrupted.
    Returns True if valid, False if corrupted.
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            # Test if the ZIP file can be read
            bad_file = zip_file.testzip()
            if bad_file is not None:
                return False
            # Check for required PPTX structure
            if '[Content_Types].xml' not in zip_file.namelist():
                return False
        return True
    except (zipfile.BadZipFile, EOFError, FileNotFoundError):
        return False


def get_document_loader(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(file_path)

    if extension == ".txt":
        from langchain_community.document_loaders import TextLoader

        return TextLoader(file_path)

    if extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(file_path)

    if extension == ".doc":
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader

        return UnstructuredWordDocumentLoader(file_path)

    if extension in (".ppt", ".pptx"):
        from langchain_community.document_loaders import UnstructuredPowerPointLoader

        return UnstructuredPowerPointLoader(file_path)

    return None


def load_documents(folder_path="documents") -> Tuple[list, dict]:
    """
    Load all supported documents from the documents folder.
    
    Returns:
        Tuple containing:
        - documents: list of loaded documents
        - errors: dict with file names as keys and error messages as values
    """
    documents = []
    errors = {}

    # iterate over files in folder
    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(file_path)[1].lower()

        # Validate PPTX files before loading
        if extension in (".ppt", ".pptx"):
            if not is_valid_pptx(file_path):
                errors[file] = "File is corrupted or invalid. Please re-upload."
                continue

        loader = get_document_loader(file_path)

        if loader is None:
            continue

        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            errors[file] = f"{type(e).__name__}: {str(e)}"
            continue

    return documents, errors