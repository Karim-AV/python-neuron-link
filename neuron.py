import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# Функция для извлечения признаков из csv-файла
def extract_features_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    X = data  # Используем все столбцы в качестве признаков
    X = X.astype('float64')

    return X

# Загрузка csv-файла и извлечение признаков
csv_file = "wireshark.csv"
X = extract_features_from_csv(csv_file)

# Определение нейронной сети
model = tf.keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

# Компиляция модели
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Обучение модели
model.fit(X, epochs=50, batch_size=32)

# Предсказания модели
predictions = model.predict(X)
for prediction in predictions:
    if prediction > 0.5:
        print("Attack detected")
    else:
        print("Normal traffic")
