import os
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup
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

# Process document files
def process_document(file_path: str) -> List[Document]:
    """Processes a file (PDF, CSV, or HTML) and converts it into text chunks."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        text = " ".join(doc.page_content for doc in docs)
    elif ext == ".csv":
        df = pd.read_csv(file_path, encoding="utf-8")
        text = " ".join(df.astype(str).stack())
    elif ext == ".html":
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents([Document(page_content=text)])

# Embed documents and store them in ChromaDB
def embed_and_store_documents(file_path: str):
    """Embeds text chunks from a file and stores them in ChromaDB."""
    documents = process_document(file_path)
    for doc in documents:
        text = doc.page_content
        embedding = embed_model.encode(text)
        collection.add(documents=[text], metadatas=[{"source": file_path}], embeddings=[embedding])

# Query ChromaDB using cosine similarity
def query_database_and_generate_answer(question: str, top_k: int = 5):
    """Queries the ChromaDB using cosine similarity and generates an answer."""
    # Embed the question
    question_embedding = embed_model.encode([question])

    # Retrieve all stored embeddings and texts
    stored_embeddings = collection.get()["embeddings"]
    stored_texts = collection.get()["documents"]

    # Compute cosine similarities
    similarities = cosine_similarity(question_embedding, stored_embeddings)[0]

    # Rank chunks by similarity
    ranked_indices = similarities.argsort()[::-1][:top_k]
    ranked_chunks = [(stored_texts[i], similarities[i]) for i in ranked_indices]

    # Print top-ranked chunks and scores
    print("\nTop-ranked Chunks:")
    for i, (chunk, score) in enumerate(ranked_chunks, start=1):
        print(f"Rank {i}: Similarity: {score:.4f}\nChunk: {chunk}\n")

    # Combine top chunks as context
    context = " ".join(chunk for chunk, _ in ranked_chunks)
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    # Generate the final answer
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = generative_model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated Answer:")
    print(generated_answer)

# Discover files in a directory
def get_file_paths(directory: str) -> List[str]:
    """Finds all PDF, CSV, and HTML files in the specified directory."""
    supported_extensions = {".pdf", ".csv", ".html"}
    file_paths = [
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if os.path.splitext(file_name)[1].lower() in supported_extensions
    ]
    return file_paths

# Example usage
if __name__ == "__main__":
    # Discover and embed files
    directory = "path_to_directory"  # Replace with your directory path
    file_paths = get_file_paths(directory)
    
    if not file_paths:
        print("No valid files found in the directory.")
    else:
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            embed_and_store_documents(file_path)

    # User interaction
    user_question = input("Enter your question: ")
    query_database_and_generate_answer(user_question)

'''
Key Changes:
Cosine Similarity:

Added cosine_similarity from sklearn.metrics.pairwise.
Used to compute similarity scores between the question and all stored embeddings.
Ranking:

Sorted chunks by similarity score in descending order.
Displayed the top k results along with their scores.
Generative Answer:

Combined top chunks as context and generated a final answer using the generative model.
Clear Output:

Displayed ranked chunks and their similarity scores for transparency.
Example Workflow:
Embed Files:

Place UTF-8 encoded files in the specified directory.
The script will embed them into ChromaDB.
Ask a Question:

Provide a question.
The system will rank chunks by cosine similarity and generate an answer.
'''