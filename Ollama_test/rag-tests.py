import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import fitz  # PyMuPDF for PDF text extraction
import torch

# Step 1: Extract text from all PDFs in a folder
def extract_text_from_pdfs(folder_path):
    """Extract text from all PDF files in a folder."""
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            documents.append({"id": file_name, "content": text, "metadata": {"source": file_name}})
    return documents

# Step 2: Load CSV data and preprocess
def load_csv_data(csv_path):
    """Load and preprocess the CSV file with question-answer pairs."""
    csv_data = pd.read_csv(csv_path)
    return [{"id": f"csv_{i}", "content": row["question"] + " " + row["answer"], "metadata": {}} for i, row in csv_data.iterrows()]

# Step 3: Set up Chroma DB and store embeddings
def setup_chroma_db(documents, embedding_model_name="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"):
    """Initialize Chroma DB and store document embeddings."""
    # Define embedding function
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
    # Initialize Chroma DB client
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="db"  # Directory to store the database
    ))
    # Create or load a collection
    collection = client.get_or_create_collection(name="qa_collection", embedding_function=embedding_function)
    # Add documents to the collection
    for doc in documents:
        collection.add(
            ids=[doc["id"]],
            documents=[doc["content"]],
            metadatas=[doc["metadata"]]
        )
    return client, collection

# Step 4: Query Chroma DB for relevant documents and print similarity coefficients
def retrieve_relevant_docs_with_scores(query, collection, top_k=3):
    """Retrieve the most relevant documents and their similarity scores for a given query."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    documents = results["documents"][0]
    scores = results["distances"][0]  # Similarity scores (lower is better for cosine distance)
    # Print documents with similarity scores
    print("\nRelevant Documents and Similarity Scores:")
    for doc, score in zip(documents, scores):
        print(f"Document: {doc[:100]}... | Similarity Score: {score:.4f}")
    return documents, scores

# Step 5: Generate an answer solely from the retrieved context
def generate_answer_with_bggpt(query, context, model_path=r"C:\Users\Owner\OneDrive\Desktop\Models\Insait9B-Q4KM\BgGPT-Gemma-2-9B-IT-v1.0.Q4_K_M.gguf"):
    """
    Generate an answer using a local INSAIT BG-GPT model, based only on the retrieved context.
    """
    # Load the tokenizer and model from the local path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    # Combine query and context into a strict prompt
    prompt = f"Въпрос: {query}\nКонтекст: {context}\nОтговор (само въз основа на контекста):"
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate response
    outputs = model.generate(
        inputs["input_ids"].to(model.device),
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main Function
if __name__ == "__main__":
    # File paths
    pdf_folder = "documents"  # Folder containing PDF files
    csv_path = "data.csv"  # Replace with your CSV file path

    # Step 1: Extract text from all PDFs in the folder
    pdf_documents = extract_text_from_pdfs(pdf_folder)
    print(f"Extracted text from {len(pdf_documents)} PDF files.")

    # Step 2: Load and preprocess CSV data
    csv_documents = load_csv_data(csv_path)
    documents = pdf_documents + csv_documents
    print("CSV data loaded and combined with PDF documents.")

    # Step 3: Set up Chroma DB and store embeddings
    client, collection = setup_chroma_db(documents)
    print("Chroma DB setup complete and embeddings stored.")

    # Step 4: Query and retrieve relevant documents with similarity scores
    query = "Как да подам данъчна декларация?"  # Replace with your query in Bulgarian
    retrieved_docs, similarity_scores = retrieve_relevant_docs_with_scores(query, collection)
    print("Relevant documents retrieved.")

    # Step 5: Generate an answer
    context = " ".join(retrieved_docs)  # Combine retrieved documents into context
    answer = generate_answer_with_bggpt(query, context)
    print("\nGenerated Answer:")
    print(answer)
