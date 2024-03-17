import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model("model.keras")

# Load the new dataset
with open('intents.json') as file:
    data = json.load(file)

# Extract patterns and labels
patterns, labels = [], []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['tag'])

# Preprocess data
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, padding='post')


# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)
encoded_labels = label_encoder.transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Function to get a response
def get_response(message):
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=padded_sequences.shape[1], padding='post')
    prediction = model.predict(padded)
    intent = label_encoder.inverse_transform([np.argmax(prediction)])
    responses = next(item['responses'] for item in data['intents'] if item['tag'] == intent[0])
    return [intent[0], np.random.choice(responses)]

text=""
while text!="exit":
    print()
    text = str(input("query >> "))
    output = get_response(text)
    print("\n Predicted Tag: ", output[0])
    print("\n bot advise >> ", output[1])