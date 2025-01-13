# streamlit run app.py
import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks."""
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    try:
        temp_file.write(uploaded_file.read())
        temp_file.close()  # Close the file handle

        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        return text_splitter.split_documents(docs)
    finally:
        # Ensure the file is deleted, even if an error occurs
        os.unlink(temp_file.name)


if __name__ == "__main__":
      
    with st.sidebar:
        st.set_page_config(page_title="RAG Question Answer")
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "⚡️ Process",
        )

    if uploaded_file and process:
        all_splits = process_document(uploaded_file)
        st.write(all_splits)