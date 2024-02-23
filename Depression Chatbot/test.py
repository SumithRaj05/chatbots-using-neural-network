

import yaml
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

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

# patterns = []
# responses = []
# tags = []
# for intent in data['intents']:
#     for pattern in intent['patterns']:
#         patterns.append(pattern)
#         responses.append(intent['responses'][0])  # Take the first response for simplicity
#         tags.append(intent['tag'])

# Tokenizing the patterns
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Encoding the labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(responses)
y = to_categorical(y)

# Neural network model
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=80, batch_size=5, verbose=1)
print(len(X))
# Save model weights
model.save("model.keras")

print("Model saved to disk.")
