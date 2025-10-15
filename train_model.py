import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


FEATURES_PATH = 'features'

def load_features_labels(features_path):
    sequences = []
    labels = []
    for phrase in os.listdir(features_path):
        phrase_path = os.path.join(features_path, phrase)
        if not os.path.isdir(phrase_path):
            continue
        for file in os.listdir(phrase_path):
            if file.endswith('.npy'):
                feature_path = os.path.join(phrase_path, file)
                features = np.load(feature_path)
                sequences.append(features)
                labels.append(phrase)
    return sequences, labels

# Load features and labels
sequences, labels = load_features_labels(FEATURES_PATH)
print(f"Loaded {len(sequences)} feature samples.")

# Pad sequences to the maximum video length
max_seq_len = max(seq.shape[0] for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')

# Encode label classes
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
print(f"Classes: {le.classes_}")

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Build a simple LSTM model
model = Sequential([
    Masking(mask_value=0., input_shape=(max_seq_len, sequences[0].shape[1])),
    LSTM(64, return_sequences=False),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=8)

# Save the trained model and label classes
model.save('sign_language_model.h5')
np.save('label_classes.npy', le.classes_)

print("Training complete and model saved.")
