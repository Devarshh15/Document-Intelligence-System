import streamlit as st
import os
import tempfile


st.set_page_config(
    page_title="AskMyDocs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

hide_streamlit_style = """
<style>
#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add custom CSS for sticky header
sticky_header_css = """
<style>
    [data-testid="stHeader"] {
        background-color: white;
        position: sticky;
        top: 0;
        z-index: 999;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
    
    [data-testid="stAppViewContainer"] {
        padding-top: 0;
    }
    
    .logo-title-container {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 1rem 1rem;
        background-color: white;
        position: sticky;
        top: 0;
        z-index: 999;
        border-bottom: 1px solid #e0e0e0;
    }
</style>
"""

st.markdown(sticky_header_css, unsafe_allow_html=True)

# from app.evaluation import evaluate_rag

from app.ingest import SUPPORTED_EXTENSIONS
from app.rag_pipeline import RAGPipeline

# Display logo and title in a custom layout (sticky header)
st.markdown('<div class="logo-title-container">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 8], gap="small")

with col1:
    # Display company logo if it exists - LARGER SIZE
    logo_path = "assets/company_logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
    else:
        st.info("📌 Place your company_logo.png in the 'assets' folder")

with col2:
    st.title("📄 AskMyDocs")
    st.caption("AI that answers from your documents, not assumptions.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# Initialize pipeline
if "documents_folder" not in st.session_state:
    st.session_state.documents_folder = tempfile.mkdtemp(prefix="askmydocs_")

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(documents_folder=st.session_state.documents_folder)


rag = st.session_state.rag


# ---------------------------
# Upload Documents
# ---------------------------

st.header("Upload Documents")
st.info(
    "Upload one or more documents and ask questions in natural language. "
    "AskMyDocs provides accurate, context-grounded answers using only the content of your uploaded documents."
)

uploaded_files = st.file_uploader(
    "Upload PDF, TXT, Word, or PowerPoint files",
    type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True
)


if uploaded_files:

    for file in uploaded_files:

        os.makedirs(st.session_state.documents_folder, exist_ok=True)

        save_path = os.path.join(
            st.session_state.documents_folder,
            os.path.basename(file.name)
        )

        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

    st.success("Documents uploaded successfully!")

    with st.spinner("Processing documents..."):
        try:
            rag.build_index()
            
            # Display any load errors
            if hasattr(rag, 'load_errors') and rag.load_errors:
                with st.expander("⚠️ File Processing Warnings", expanded=True):
                    for filename, error in rag.load_errors.items():
                        st.warning(f"**{filename}**: {error}")
            
            # Show success message if documents were loaded
            if rag.vector_store is not None:
                st.success("✅ Index built successfully! Ready to answer questions.")
            else:
                st.error("❌ No valid documents were processed. Please upload valid PDF, Word, Text, or PowerPoint files.")
        except Exception as e:
            st.error(f"❌ Error processing documents: {type(e).__name__}: {str(e)}")
            with st.expander("📋 Technical Details"):
                st.code(str(e))


# ---------------------------
# Ask Questions
# ---------------------------

st.header("Ask Question")

question = st.text_input("Enter your question", key="question_input")

if st.button("Clear Question Cache"):
    for key in ["question_input", "last_answer", "last_sources"]:
        st.session_state.pop(key, None)

    # Clear any Streamlit caches if present.
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Question cache cleared.")
    st.rerun()


if st.button("Ask"):
    if question and question.strip():
        with st.spinner("Generating answer..."):
            try:
                answer, sources = rag.ask(question.strip())

                st.session_state.last_answer = answer
                st.session_state.last_sources = sources

                st.subheader("Answer")
                st.success(answer)

                st.subheader("Sources")
                if sources:
                    for source in sources:
                        st.write(f"- {source}")
                else:
                    st.info("No supporting source files were identified.")

            except Exception as error:
                # Detailed diagnostic goes to server logs.
                print(
                    f"AskMyDocs question failed: "
                    f"{type(error).__name__}: {error}"
                )
                st.error(
                    "Unable to generate an answer right now. "
                    "Please try again later. If the issue persists, "
                    "check the app logs."
                )
    else:
        st.warning("Please enter a question before asking.")