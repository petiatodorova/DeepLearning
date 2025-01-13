from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a pre-trained model and tokenizer
model_name = "rmihaylov/bert-base-squad-theseus-bg"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example text
text = "I love machine learning!"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Pass the tokenized input to the model
outputs = model(**inputs)

# View the model's output (e.g., logits for classification)
print(outputs)
