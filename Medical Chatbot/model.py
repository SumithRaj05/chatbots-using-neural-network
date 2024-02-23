import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from numpy import random

# Load the intents JSON data
with open('intents.json') as file:
    data = json.load(file)

# Extracting features and labels from the loaded data
patterns = []
responses = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'][0])  # Take the first response for simplicity
        tags.append(intent['tag'])

# Tokenizing the patterns
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Encoding the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)
y = to_categorical(y)

# Load the saved model
model = load_model("model.keras")

# Function to predict intent
def predict_intent(text):
    # Preprocess input text
    text_vectorized = vectorizer.transform([text]).toarray()
    # Perform inference
    prediction = model.predict(text_vectorized)
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    # Get the tag associated with the predicted class
    tag = label_encoder.inverse_transform([predicted_class])[0]
    responses_list = [intent['responses'][0] for intent in data['intents'] if intent['tag'] == tag]
    response = random.choice(responses_list)
    return [tag, response]

# Example usage
user_input=""
while user_input!="exit":
    print()
    user_input = str(input("query >> "))
    predicted_intent = predict_intent(user_input)
    print("-------------------------prediction-----------------------")
    print("PREDICTED TAG: ", predicted_intent[0])
    print("\nresponse >> ", predicted_intent[1])

