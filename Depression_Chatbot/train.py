import yaml
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load dataset
with open('depression.yml', 'r') as file:
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

# Split data (this example uses all data for training for simplicity)
X_train, y_train = padded_sequences, categorical_labels

# Define neural network architecture
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=padded_sequences.shape[1]),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model

print("\nRecommended epoches: ", len(X_train), "\n")
model.fit(X_train, y_train, epochs=70, verbose=1)

model.save("model.keras")

print("\nmodel saved in model.keras!")