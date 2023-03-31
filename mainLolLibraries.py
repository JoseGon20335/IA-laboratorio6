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

def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# PARTE 1.2
print("Loading dataset...")
data = load_csv("high_diamond_ranked_10min.csv")

train_val, test = train_test_split(data, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1, random_state=42)

train = pd.DataFrame(train, columns=data.columns)
val = pd.DataFrame(val, columns=data.columns)

features = ['blueKills', 'blueTowersDestroyed', 'blueTotalGold', 'blueTotalMinionsKilled',
            'redKills', 'redTowersDestroyed', 'redTotalGold', 'redTotalMinionsKilled']
target = 'blueWins'

X_train = train.loc[:,features]
y_train = train.loc[:,target]
X_val = val.loc[:,features]
y_val = val.loc[:,target]
X_test = test.loc[:,features]
y_test = test.loc[:,target]

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=features,
          class_names=["Red Wins", "Blue Wins"])
plt.show()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión de las predicciones:", accuracy)
