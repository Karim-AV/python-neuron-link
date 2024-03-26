import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#Import Dataset
df = pd.read_csv("KDDTest.csv")

# Определение признаков и целевой переменной

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Кодирование категориальных признаков
encoder = LabelEncoder()
X_train['protocol_type'] = encoder.fit_transform(X_train['protocol_type'])
X_train['service'] = encoder.fit_transform(X_train['service'])
X_train['flag'] = encoder.fit_transform(X_train['flag'])

X_test['protocol_type'] = encoder.transform(X_test['protocol_type'])
X_test['service'] = encoder.transform(X_test['service'])
X_test['flag'] = encoder.transform(X_test['flag'])

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание модели MLP
model = Sequential([
    Dense(128, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# Оценка модели
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy}')

# Предсказание на тестовом наборе
y_pred = model.predict_classes(X_test_scaled)

# Вывод отчета о классификации
print(classification_report(y_test, y_pred))