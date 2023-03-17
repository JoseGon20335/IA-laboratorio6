from math import log
import random

# Load the dataset
print("Loading dataset...")
with open('high_diamond_ranked_10min.csv', 'r') as f:
    data2 = f.readlines()

# Remove the header row
header = data2[0].strip().split(',')
data = data2[1:]
X = []
y = []
for row in data:
    row_data = row.strip().split(',')
    print(row_data[-1])
    X.append([float(x) if x.isdigit() else x for x in row_data[:-1]])
    y.append(float(row_data[-1]))

print('x', X)
print('y', y)
print("Loaded", len(y), "samples with", len(X[0]), "features")

# Encoding the categorical variables
categorical_vars = [i for i in range(
    len(header)) if header[i] == 'categorical_variable']
if len(categorical_vars) > 0:
    print("Encoding", len(categorical_vars), "categorical variables...")
for i in categorical_vars:
    categories = list(set([row[i] for row in X]))
    for j in range(len(X)):
        X[j][i] = categories.index(X[j][i])
print("Done encoding categorical variables")

# Checking if the dataset is balanced and applying random oversampling if it's not
n_samples = len(y)
n_classes = len(set(y))
if y and min([y.count(c) for c in range(n_classes)]) < n_samples/n_classes:
    print("Dataset is unbalanced, applying random oversampling...")
    max_class_samples = max([y.count(c) for c in range(n_classes)])
    for c in range(n_classes):
        class_samples = [X[i] for i in range(n_samples) if y[i] == c]
        while len(class_samples) < max_class_samples:
            if class_samples:
                class_samples.append(
                    class_samples[random.randint(0, len(class_samples)-1)])
            else:
                break
            y.append(c)
            X.append(class_samples[-1])
    print("Done oversampling, new dataset size is", len(y))

# Scaling the variables
numeric_vars = [i for i in range(
    len(header)) if header[i].startswith('numeric_variable')]
if len(numeric_vars) > 0:
    print("Scaling", len(numeric_vars), "numeric variables...")
for i in numeric_vars:
    col_values = [row[i] for row in X]
    min_value = min(col_values)
    max_value = max(col_values)
    for j in range(len(X)):
        X[j][i] = (X[j][i] - min_value) / (max_value - min_value)
print("Done scaling numeric variables")

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
