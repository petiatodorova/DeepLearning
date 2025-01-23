from sentence_transformers import SentenceTransformer
import os

# Define the directory to save the model
save_path = r"C:\Users\Owner\OneDrive\Desktop\MiniLM12"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Download and load the model
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# Save the model to the specified directory
model.save(save_path)

print(f"Model saved to: {save_path}")

# use it
# model = SentenceTransformer(r"C:\Users\Owner\OneDrive\Desktop\DeepLearning\EmbeddingModel")

