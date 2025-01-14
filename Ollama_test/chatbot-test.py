import os
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForCausalLM

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

# Helper functions
def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with pdfplumber.open(file_path) as pdf:
        text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def extract_text_from_csv(file_path):
    """Extract text from a CSV file."""
    df = pd.read_csv(file_path)
    return " ".join(df.astype(str).stack())

def extract_text_from_html(file_path):
    """Extract text from an HTML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text()

def embed_and_store(file_path):
    """Embed text from a file and store in ChromaDB."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".csv":
        text = extract_text_from_csv(file_path)
    elif ext == ".html":
        text = extract_text_from_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Split text into chunks
    embeddings = embed_model.encode(chunks)  # Embed chunks
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(documents=[chunk], metadatas=[{"chunk_id": i}], embeddings=[embedding])

# Load generative model
generative_model_name = "INSAIT/BGGPT"
tokenizer = AutoTokenizer.from_pretrained(generative_model_name)
generative_model = AutoModelForCausalLM.from_pretrained(generative_model_name)

def generate_best_answer(question, context):
    """Generate a refined answer using a generative model."""
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = generative_model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Query function
def query_database_and_generate_answer(question, top_k=5):
    """Query ChromaDB and use the generative model to refine the answer."""
    question_embedding = embed_model.encode([question])
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)
    
    # Print top-ranked chunks and scores
    for i, (doc, score) in enumerate(zip(results["documents"], results["distances"])):
        print(f"Rank {i+1}: Score: {score}\nChunk: {doc}\n")
    
    # Combine top chunks as context
    context = " ".join(results["documents"])
    generated_answer = generate_best_answer(question, context)
    
    print("\nGenerated Answer:")
    print(generated_answer)

# Embed files into the database
file_paths = [
    "example.pdf",  # Replace with your file paths
    "example.csv",
    "example.html"
]
for file_path in file_paths:
    embed_and_store(file_path)

# User interaction
user_question = input("Enter your question: ")
query_database_and_generate_answer(user_question)
