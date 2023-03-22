import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from math import log
import random
from utils import load_csv
from tree import tree

# Load the dataset
print("Loading dataset...")
data = load_csv("high_diamond_ranked_10min.csv")

train_set = []
validation_set = []
test_set = []
train_size = int(len(data) * 0.8)
validation_size = int(len(data) * 0.1)
random.shuffle(data)
for i in range(train_size):
    train_set.append(data[i])
for i in range(train_size, train_size+validation_size):
    validation_set.append(data[i])
for i in range(train_size+validation_size, len(data)):
    test_set.append(data[i])

# PARTE 2.1

tree = tree()
print("Building tree...")
print("Training set size:", len(train_set))
print('tranining set:', train_set)
tree_model = tree.build_tree(train_set)
actual = [row[-1] for row in validation_set]
predicted = [tree.predict(tree_model, row) for row in validation_set]
precision = 0
for i in range(len(actual)):
    if actual[i] == predicted[i]:
        precision += 1
precision = precision / float(len(actual))

print('PRECISION VALIDATION MODEL:', precision*100)

actual = [row[-1] for row in test_set]
predicted = [tree.predict(tree_model, row) for row in test_set]
precision = 0
print('ACTUAL:')
for i in range(len(actual)):
    if actual[i] == predicted[i]:
        precision += 1
precision = precision / float(len(actual))
print('PRECISION TEST MODEL:', precision*100)

# PARTE 2.2

# Divida los datos en conjuntos de entrenamiento, validación y prueba
train_val, test = train_test_split(data, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1, random_state=42)

# Seleccione las características y la variable objetivo
features = ['blueKills', 'blueTowersDestroyed', 'blueTotalGold', 'blueTotalMinionsKilled',
            'redKills', 'redTowersDestroyed', 'redTotalGold', 'redTotalMinionsKilled']
target = 'blueWins'

# Separe las características y la variable objetivo en conjuntos de entrenamiento, validación y prueba
X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
y_test = test[target]

# Cree un modelo de Árbol de Decisión utilizando el conjunto de entrenamiento
model = DecisionTreeClassifier(random_state=42)

# Ajuste los parámetros del modelo utilizando el conjunto de validación
model.fit(X_train, y_train)

# Mida la precisión del modelo utilizando el conjunto de prueba
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Grafique el árbol de decisión para visualizar cómo se toman las decisiones en el modelo
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=features,
          class_names=["Red Wins", "Blue Wins"])
plt.show()

# Realice predicciones con el modelo y mida la precisión de las predicciones utilizando el conjunto de prueba
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión de las predicciones:", accuracy)
