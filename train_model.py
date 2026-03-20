import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_PATH = "dataset"
actions = np.array(['eating','drinking','walking','sitting','waving'])

sequences = []
labels = []

for idx, action in enumerate(actions):

    action_path = os.path.join(DATA_PATH, action)

    for file in os.listdir(action_path):

        data = np.load(os.path.join(action_path, file))

        # Only keep sequences with correct shape
        if data.shape == (30, 99):
            sequences.append(data)
            labels.append(idx)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,99)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30)

model.save("action_model.h5")

print("✅ Model training completed!")