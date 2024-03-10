import json
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from scapy.layers.inet import IP

# Load the  file
with open('atack.json', 'r') as f:
    data = json.load(f)

# Extract features and create 'target' column

for packet in packets:
    features = {
        'src_ip': packet[IP].src,
        'dest_ip': packet[IP].dst,
        'protocol': packet[IP].proto,
        'length': len(packet),
    }

    # Assuming you have a threshold for the 'target' column
    data.append({'features': features, 'target': 1 if len(packet) > threshold else 0})

# Convert data to DataFrame
df = pd.DataFrame(data)

# Prepare data for model training
X = pd.DataFrame(df['features'].tolist())
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define neural network architecture - a sequence of layers
model = tf.keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

# Visualization of the neural network in TensorFlow
tensorboard = TensorBoard(log_dir="logs")

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[tensorboard])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Make predictions
predictions = model.predict(X_test)
for prediction in predictions:
    if prediction > 0.5:
        print("Attack detected")
    else:
        print("Normal traffic")
