import os
import tempfile
from typing import List
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.docstore.document import Document
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize ChromaDB client
client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Load embedding model
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model = SentenceTransformer(embedding_model_name)

# Set up Chroma collection
collection_name = "bulgarian_qa"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
)

# Generative model (INSAIT BGGPT)
generative_model_name = "INSAIT/BGGPT"
tokenizer = AutoTokenizer.from_pretrained(generative_model_name)
generative_model = AutoModelForCausalLM.from_pretrained(generative_model_name)

# Function to process documents
def process_document(file_path: str) -> List[Document]:
    """Processes a document file by converting it to text chunks."""
    loader = PyMuPDFLoader(file_path)  # Load the PDF file
    docs = loader.load()  # Extract text from the PDF
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

# Embed documents and store them in ChromaDB
def embed_and_store_documents(file_path: str):
    """Embeds text chunks from a file and stores them in ChromaDB."""
    documents = process_document(file_path)
    for doc in documents:
        text = doc.page_content
        embedding = embed_model.encode(text)
        collection.add(documents=[text], metadatas=[{"source": file_path}], embeddings=[embedding])

# Query ChromaDB and generate an answer
def query_database_and_generate_answer(question: str, top_k: int = 5):
    """Queries the ChromaDB and generates an answer using a generative model."""
    question_embedding = embed_model.encode([question])
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)
    
    # Print top-ranked chunks and scores
    for i, (doc, score) in enumerate(zip(results["documents"], results["distances"])):
        print(f"Rank {i+1}: Score: {score}\nChunk: {doc}\n")
    
    # Combine top chunks as context
    context = " ".join(results["documents"])
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = generative_model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated Answer:")
    print(generated_answer)

# Example usage
if __name__ == "__main__":
    # Embed documents
    file_paths = [
        "example.pdf"  # Replace with your file paths
    ]
    for file_path in file_paths:
        embed_and_store_documents(file_path)

    # User interaction
    user_question = input("Enter your question: ")
    query_database_and_generate_answer(user_question)
