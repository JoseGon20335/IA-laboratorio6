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

traininPorcentage = 0.8
validationPorcentage = 0.1

train, val, test = [], [], []
trainSize = int(len(data) * traininPorcentage)
valSize = int(len(data) * validationPorcentage)
random.shuffle(data)
for i in range(trainSize):
    train.append(data[i])
for i in range(trainSize, trainSize+valSize):
    val.append(data[i])
for i in range(trainSize+valSize, len(data)):
    test.append(data[i])

# PARTE 1.1

tree = tree()
print("Building tree...")
print("Training set size:", len(train))
print('tranining set:', train)
tree_model = tree.buildTree(train)
actual = [row[-1] for row in val]
predicted = [tree.predict(row, tree_model) for row in val]
precision = 0
for i in range(len(actual)):
    if actual[i] == predicted[i]:
        precision += 1
precision = precision / float(len(actual))

precisionResult = precision*100*100

print('PRECISION VALIDATION MODEL:', precisionResult)

actual = [row[-1] for row in test]
predicted = [tree.predict(row, tree_model) for row in test]
precision = 0
print('ACTUAL:')
for i in range(len(actual)):
    if actual[i] == predicted[i]:
        precision += 1
precision = precision / float(len(actual))

precisionResult = precision*100*100

print('PRECISION TEST MODEL:', precisionResult)
