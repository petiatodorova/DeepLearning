import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or PDF file", type=["csv", "pdf"])

if uploaded_file is not None:
    # Handle CSV files
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV file:")
        st.dataframe(df)

    # Handle PDF files
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.write("Extracted text from uploaded PDF:")
        st.text_area("PDF Content", text, height=300)

    else:
        st.warning("Unsupported file type. Please upload a CSV or PDF.")
