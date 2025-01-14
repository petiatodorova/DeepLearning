from sentence_transformers import SentenceTransformer
import os

# use it
# model = SentenceTransformer(r"C:\Users\Owner\OneDrive\Desktop\DeepLearning\EmbeddingModel")
# Define the directory to save the model
path_to_model = r"C:\Users\Owner\OneDrive\Desktop\Models"


def download_model(path_to_model):

    # Create the directory if it doesn't exist
    os.makedirs(path_to_model, exist_ok=True)

    # Download and load the model
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)

    # Save the model to the specified directory
    model.save(path_to_model)

    print(f"Model saved to: {path_to_model}")


def use_model(path_to_model):

    # Load the model
    model = SentenceTransformer(path_to_model)

    # Test the model
    sentence = "This is a test sentence."
    embedding = model.encode(sentence)
    print("Embedding:", embedding)

    # Print the embedding's shape and the first few elements
    print(f"Embedding dimensions: {len(embedding)}")
    print("First 10 values of the embedding:", embedding[:10])  # Show the first 10 values for example


if __name__ == "__main__":
    download_model(path_to_model)
    use_model(path_to_model)