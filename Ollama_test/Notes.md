Model:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

C:\Users\Owner\.cache

ollama pull paraphrase-multilingual

https://www.youtube.com/watch?v=1y2TohQdNbo

C:\Users\Owner\OneDrive\Desktop\DeepLearning\EmbeddingModel

https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

sentences = [
"The weather is lovely today.",
"It's so sunny outside!",
"He drove to the stadium."
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)

# [3, 3]

---

nomic-embed-text from ollama
