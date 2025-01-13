from sentence_transformers import CrossEncoder

# Load a multilingual CrossEncoder model
model = CrossEncoder('xlm-roberta-base', device='cuda' if torch.cuda.is_available() else 'cpu')

# Example Bulgarian sentences
sentence1 = "Обичам машинното обучение."
sentence2 = "Машинното обучение е моята страст."

# Compute the similarity score between the sentences
score = model.predict([(sentence1, sentence2)])[0]
print(f"Similarity score: {score}")
