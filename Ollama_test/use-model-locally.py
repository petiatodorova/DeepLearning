from sentence_transformers import SentenceTransformer

# Path to the local directory where the model is saved
local_model_path = r"C:\Users\Owner\OneDrive\Desktop\Models\MiniLM12"

# Load the model
model = SentenceTransformer(local_model_path)

# Test the model
sentence = "This is a test sentence."
embedding = model.encode(sentence)
print("Embedding:", embedding)

# Print the embedding's shape and the first few elements
print(f"Embedding dimensions: {len(embedding)}")
print("First 10 values of the embedding:", embedding[:10])  # Show the first 10 values for example