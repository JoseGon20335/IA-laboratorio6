# Load the dataset
from math import log
import random
with open('high_diamond_ranked_10min.csv', 'r') as f:
    data = f.readlines()

# Remove the header row
header = data[0].strip().split(',')
data = data[1:]

# Split the data into X and y
X = []
y = []
for row in data:
    row_data = row.strip().split(',')
    if row_data[-1].isdigit():
        X.append([float(x) if x.isdigit() else x for x in row_data[:-1]])
        y.append(int(row_data[-1]))


# Encoding the categorical variables
categorical_vars = [i for i in range(
    len(header)) if header[i] == 'categorical_variable']
for i in categorical_vars:
    categories = list(set([row[i] for row in X]))
    for j in range(len(X)):
        X[j][i] = categories.index(X[j][i])

# Checking if the dataset is balanced and applying random oversampling if it's not
n_samples = len(y)
n_classes = len(set(y))
if min([y.count(c) for c in range(n_classes)]) < n_samples/n_classes:
    max_class_samples = max([y.count(c) for c in range(n_classes)])
    for c in range(n_classes):
        class_samples = [X[i] for i in range(n_samples) if y[i] == c]
        while len(class_samples) < max_class_samples:
            class_samples.append(
                class_samples[random.randint(0, len(class_samples)-1)])
            y.append(c)
            X.append(class_samples[-1])

# Scaling the variables
numeric_vars = [i for i in range(
    len(header)) if header[i].startswith('numeric_variable')]
for i in numeric_vars:
    col_values = [row[i] for row in X]
    min_value = min(col_values)
    max_value = max(col_values)
    for j in range(len(X)):
        X[j][i] = (X[j][i] - min_value) / (max_value - min_value)

# Variable selection using SelectKBest and f_classif


def entropy(y):
    p1 = y.count(1)/len(y)
    p0 = 1 - p1
    if p0 == 0 or p1 == 0:
        return 0
    else:
        return -p0*log(p0, 2) - p1*log(p1, 2)


def information_gain(X, y, i):
    col_values = [row[i] for row in X]
    col_thresholds = sorted(list(set(col_values)))
    max_gain = 0
    best_threshold = None
    for j in range(len(col_thresholds)-1):
        threshold = (col_thresholds[j] + col_thresholds[j+1]) / 2
        left_y = [y[k] for k in range(len(y)) if X[k][i] <= threshold]
        right_y = [y[k] for k in range(len(y)) if X[k][i] > threshold]
        left_entropy = entropy(left_y)
        right_entropy = entropy(right_y)
        gain = entropy(y) - (len(left_y)/len(y)*left_entropy +
                             len(right_y)/len(y)*right_entropy)
        if gain > max_gain:
            max_gain = gain
            best_threshold = threshold
    return max_gain, best_threshold


k = 5
feature_scores = []
for i in range(len(header)-1):
    if i in categorical_vars:
        score = 0
    else:
        score, _ = information_gain
