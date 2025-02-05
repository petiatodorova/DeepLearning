import os
import tensorflow as tf
from transformers import pipeline

# Turn off oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the pipeline for question answering
pipe = pipeline("question-answering", model="rmihaylov/bert-base-squad-theseus-bg")

# Define the question and context
question = "Кой е най-големият град?"
# question = "Кои са най-големите градове?"
# question = "Кои са най-големите градове в България?"
# question = "Каква е площта на България?"
# question = "С кои държави граничи България?"
# question = "Коя е столицата на България?"

context = """
Република България е държава в Югоизточна Европа. 
Граничи на север с Румъния, на запад – със Сърбия и Република Северна Македония, на юг – с Гърция, на югоизток – с Турция и на изток – с Черно море. 
С площ почти 111 000 km² и население около 6 520 000 души (2021) тя е съответно на 11-о и 16-о място в Европейския съюз. 
София е столицата и най-големият град в страната, следвана от Пловдив, Варна, Бургас и Русе. 
Най-ранните свидетелства за присъствие на хомо сапиенс по земите на днешна България датират от преди около 46 хиляди години, или епохата на палеолита.[10][11] 
Към петото хилядолетие преди н.е. в североизточна България процъфтява култура, която създава най-ранните златни украшения в Европа. 
От античността до Тъмните векове по земите на днешна България се развиват културите на траките, древните гърци, келтите, готите и римляните. 
"""

# Perform inference
result = pipe(question=question, context=context)

# Print the answer
print("Answer:", result['answer'])
print("Score:", result['score'])
print("Start position:", result['start'])
print("End position:", result['end'])
