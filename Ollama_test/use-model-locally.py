from sentence_transformers import SentenceTransformer

# Path to the local directory where the model is saved
local_model_path = r"C:\Users\Owner\OneDrive\Desktop\DeepLearning\EmbeddingModel"

# Load the model
model = SentenceTransformer(local_model_path)

# Test the model
sentence = "This is a test sentence."
embedding = model.encode(sentence)
print("Embedding:", embedding)
