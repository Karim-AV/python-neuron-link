#Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
df = pd.read_csv("KDDTest.csv")

columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'])

df.columns = columns
df.head()
df.info()
print(df.describe())
df = df.drop('level', axis=1)
print(df.attack.unique())
display(df.describe())

# Применение LabelEncoder ко всем категориальным столбцам
label_encoder = preprocessing.LabelEncoder()
df_encoded = df.apply(label_encoder.fit_transform)

#Train-Test Split
X = df_encoded.drop(labels=['attack'], axis=1)
y = df_encoded[['attack']]
print('X_train has shape:', X.shape, '\ny_train has shape:', y.shape)

# splitting the dataset 80% for training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели нейронной сети
mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 22), max_iter=100, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Предсказание на тестовом наборе данных
y_pred = mlp.predict(X_test_scaled)
# Вычисление точности
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# задача 1 Вычисление точности на каждой эпохе

# задача 2 преза по каждому блоку
