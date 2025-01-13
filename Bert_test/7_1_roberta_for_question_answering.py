"""
!pip install transformers[torch]==4.38.2
!pip install datasets===2.13.1
!pip install plotly
"""


import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from scipy.special import softmax
import plotly.express as px
import pandas as pd
import numpy as np

"""## Data Preparation 📖

Define the context and question that will be used to demonstrate the BERT model's question-answering capabilities.
"""

context = "The giraffe is a large African hoofed mammal belonging to the genus Giraffa. It is the tallest living terrestrial animal and the largest ruminant on Earth. Traditionally, giraffes were thought to be one species, Giraffa camelopardalis, with nine subspecies. Most recently, researchers proposed dividing them into up to eight extant species due to new research into their mitochondrial and nuclear DNA, as well as morphological measurements. Seven other extinct species of Giraffa are known from the fossil record."
question = "How many giraffe species are there?"

"""## Model Setup 🤖

Load the BERT model and tokenizer designed for question answering tasks.
"""

model_name = "rmihaylov/bert-base-squad-theseus-bg"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

"""## Question Answering

Tokenize the context and the question, and perform inference to get model predictions.
"""

inputs = tokenizer(question, context, return_tensors="pt")
tokenizer.tokenize(context)

# Running the model and getting the start and end scores
with torch.no_grad():
    outputs = model(**inputs)
start_scores, end_scores = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]

"""### Visualization of Token Scores 🎯

Plot the scores associated with each token to understand the model's decision-making process.
"""

# Creating a dataframe with the scores and plotting them
scores_df = pd.DataFrame({
    "Token Position": list(range(len(start_scores))) * 2,
    "Score": list(start_scores) + list(end_scores),
    "Score Type": ["Start"] * len(start_scores) + ["End"] * len(end_scores),
})
px.bar(scores_df, x="Token Position", y="Score", color="Score Type", barmode="group", title="Start and End Scores for Tokens")

"""### Extracting the Answer

Identify the tokens that comprise the answer based on the highest start and end scores.
"""

# Getting the answer from the model
start_idx = np.argmax(start_scores)
end_idx = np.argmax(end_scores)
answer_ids = inputs.input_ids[0][start_idx: end_idx + 1]
answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
answer = tokenizer.convert_tokens_to_string(answer_tokens)

"""## Advanced Usage: Sliding Windows & Stride for QA 🚀

Define a function to automate the question-answering process with new contexts and questions.
"""

# Part 2
# Defining a function to predict the answer to a question given a context
def predict_answer(context, question):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start_scores, end_scores = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]
    start_idx = np.argmax(start_scores)
    end_idx = np.argmax(end_scores)
    confidence_score = (start_scores[start_idx] + end_scores[end_idx]) /2
    answer_ids = inputs.input_ids[0][start_idx: end_idx + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if answer != tokenizer.cls_token:
        return answer, confidence_score
    return None, confidence_score

"""
### Evaluating the Function

Test the `predict_answer` function with various questions and contexts."""

# Defining a new context and predicting answers to some questions
context = """
Република България е държава в Югоизточна Европа. 
Граничи на север с Румъния, на запад – със Сърбия и Република Северна Македония, на юг – с Гърция, на югоизток – с Турция и на изток – с Черно море. 
С площ почти 111 000 km² и население около 6 520 000 души (2021) тя е съответно на 11-о и 16-о място в Европейския съюз. 
София е столицата и най-големият град в страната, следвана от Пловдив, Варна, Бургас и Русе. 
Най-ранните свидетелства за присъствие на хомо сапиенс по земите на днешна България датират от преди около 46 хиляди години, или епохата на палеолита.[10][11] 
Към петото хилядолетие преди н.е. в североизточна България процъфтява култура, която създава най-ранните златни украшения в Европа. 
От античността до Тъмните векове по земите на днешна България се развиват културите на траките, древните гърци, келтите, готите и римляните. 

"""

len(tokenizer.tokenize(context))
predict_answer(context, "What is coffee?")
predict_answer(context, "What are the most common coffee beans?")
predict_answer(context, "How can I make ice coffee?")
predict_answer(context[4000:], "How many people are dependent on coffee for their income?")

# Defining a function to chunk sentences
def chunk_sentences(sentences, chunk_size, stride):
    chunks = []
    num_sentences = len(sentences)
    for i in range(0, num_sentences, chunk_size - stride):
        chunk = sentences[i: i + chunk_size]
        chunks.append(chunk)
    return chunks

sentences = [
    "Sentence 1.",
    "Sentence 2.",
    "Sentence 3.",
    "Sentence 4.",
    "Sentence 5.",
    "Sentence 6.",
    "Sentence 7.",
    "Sentence 8.",
    "Sentence 9.",
    "Sentence 10."
]


sentences = context.split("\n")
chunked_sentences = chunk_sentences(sentences, chunk_size=3, stride=1)


questions = ["Кой е най-големият град?", "Кои са най-големите градове?", "Каква е площта на България?", "С кои държави граничи България?", "Коя е столицата на България?"]

answers = {}

for chunk in chunked_sentences:
    sub_context = "\n".join(chunk)
    for question in questions:
        answer, score = predict_answer(sub_context, question)

        if answer:
            if question not in answers:
                answers[question] = (answer, score)
            else:
                if score > answers[question][1]:
                    answers[question] = (answer, score)

print(answers)
