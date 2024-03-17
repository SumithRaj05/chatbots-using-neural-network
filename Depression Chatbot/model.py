import yaml
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load dataset
with open('Depression Chatbot\depression.yml', 'r') as file:
    dataset = yaml.safe_load(file)

# Extract conversations and responses
patterns = []
labels = []
for conv in dataset['conversations']:
    for i in range(len(conv) - 1):
        patterns.append(conv[i])
        labels.append(conv[i + 1])

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

# Load the saved model
model = load_model("model.keras")

# Function to get a response
def get_response(message):
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=padded_sequences.shape[1], padding='post')
    prediction = model.predict(padded)
    intent = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return intent

user_input=""
while user_input!="exit":
    print()
    user_input = str(input("query >> "))
    predicted_intent = get_response(user_input)
    print("\nresponse >> ", predicted_intent)

