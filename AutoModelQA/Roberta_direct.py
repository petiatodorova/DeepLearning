from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load model and tokenizer
model_name = "rmihaylov/bert-base-squad-theseus-bg"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Inputs
question = "Кои са най-големият град?"
# question = "Кои са най-големите градове?"
# question = "Кои са най-големите градове в България?"
# question = "Каква е площта на България?"
# question = "С кои държави граничи България?"
# question = "Коя е столицата на България?"

# Context
context = """
Република България е държава в Югоизточна Европа. 
Граничи на север с Румъния, на запад – със Сърбия и Република Северна Македония, на юг – с Гърция, на югоизток – с Турция и на изток – с Черно море. 
С площ почти 111 000 km² и население около 6 520 000 души (2021) тя е съответно на 11-о и 16-о място в Европейския съюз. 
София е столицата и най-големият град в страната, следвана от Пловдив, Варна, Бургас и Русе. 
Най-ранните свидетелства за присъствие на хомо сапиенс по земите на днешна България датират от преди около 46 хиляди години, или епохата на палеолита.[10][11] 
Към петото хилядолетие преди н.е. в североизточна България процъфтява култура, която създава най-ранните златни украшения в Европа. 
От античността до Тъмните векове по земите на днешна България се развиват културите на траките, древните гърци, келтите, готите и римляните. 
"""

tokens = tokenizer.tokenize(context)
print(tokens)

inputs = tokenizer(question, context, return_tensors="pt")

# Model inference
outputs = model(**inputs)

# Extract logits
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Get start and end positions
start_position = start_logits.argmax()
end_position = end_logits.argmax()

# Decode answer
answer = tokenizer.decode(inputs.input_ids[0][start_position:end_position+1])
print("Answer:", answer)


