import yaml
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from numpy import random

# Load dataset
with open('depression.yml', 'r') as file:
    dataset = yaml.safe_load(file)

# Extract conversations and responses
patterns = []
responses = []
for conv in dataset['conversations']:
    for i in range(len(conv) - 1):
        patterns.append(conv[i])
        responses.append(conv[i + 1])

# Tokenizing the patterns
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Encoding the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(responses)
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
    return tag

# Example usage
user_input=""
while user_input!="exit":
    print()
    user_input = str(input("query >> "))
    predicted_intent = predict_intent(user_input)
    print("\nresponse >> ", predicted_intent)

