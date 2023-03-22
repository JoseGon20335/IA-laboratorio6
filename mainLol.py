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

# PARTE 1.1

tree = tree()
print("Building tree...")
print("Training set size:", len(train_set))
print('tranining set:', train_set)
tree_model = tree.build_tree(train_set)
actual = [row[-1] for row in validation_set]
predicted = [tree.predict(row, tree_model) for row in validation_set]
precision = 0
for i in range(len(actual)):
    if actual[i] == predicted[i]:
        precision += 1
precision = precision / float(len(actual))

print('PRECISION VALIDATION MODEL:', precision*100)

actual = [row[-1] for row in test_set]
predicted = [tree.predict(row, tree_model) for row in test_set]
precision = 0
print('ACTUAL:')
for i in range(len(actual)):
    if actual[i] == predicted[i]:
        precision += 1
precision = precision / float(len(actual))
print('PRECISION TEST MODEL:', precision*100)
