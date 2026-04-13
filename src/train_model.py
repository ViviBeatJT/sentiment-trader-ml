import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers

def create_sequences(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        # We use Price (open, high, low, close, volume) + Sentiment
        X.append(data[i:(i + window_size), :-1]) 
        # The target is the last column
        y.append(data[i + window_size, -1])
    return np.array(X), np.array(y)

# 1. Load Data
df = pd.read_csv("data/training_master.csv")
# Use: open, high, low, close, volume, sentiment_score, target
features = df[['open', 'high', 'low', 'close', 'volume', 'sentiment_score', 'target']].values

# 2. Scale Data (TensorFlow performs best with 0 to 1 values)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# 3. Create Windows
X, y = create_sequences(scaled_features, window_size=20)

# 4. Split into Train/Test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build the Model
model = tf.keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(50, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(50),
    layers.Dropout(0.2),
    layers.Dense(25, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Sigmoid for 0 or 1 prediction
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train
print("Starting training...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 7. Save
model.save("models/sentiment_lstm_model.keras")
print("Model saved to models/ folder!")