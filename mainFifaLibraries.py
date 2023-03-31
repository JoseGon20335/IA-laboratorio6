import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

data = pd.read_csv("DataCleaned.csv", low_memory=False)

train, valtest = train_test_split(data, test_size=0.2, random_state=1)
val, test = train_test_split(valtest, test_size=0.5, random_state=1)

trainingX = train.drop(["Potential"], axis=1)
trainingY = train["Potential"]
valX = val.drop(["Potential"], axis=1)
valY = val["Potential"]
testingX = test.drop(["Potential"], axis=1)
testingY = test["Potential"]

model = DecisionTreeRegressor(random_state=1)
model.fit(trainingX, trainingY)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 características principales:")
for f in range(5):
    print("%d. %s (%f)" % (f + 1, trainingX.columns[indices[f]], importances[indices[f]]))

predictionYValue = model.predict(valX)
print("Puntaje R2 en set de validación: ", r2_score(valY, predictionYValue))

pipeline = make_pipeline(
    StandardScaler(),
    Ridge(alpha=10, max_iter=100, tol=0.0001, solver="lsqr", random_state=42),
)

pipeline.fit(trainingX, trainingY)

predictionYTest = pipeline.predict(testingX)
print("Puntaje R2 en set de testing: ", r2_score(testingY, predictionYTest))

params = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_split": [2, 5, 10, 20],
}
gridSearch = GridSearchCV(
    DecisionTreeRegressor(random_state=42), params, cv=5, n_jobs=-1
)
gridSearch.fit(pd.concat([trainingX, valX]), pd.concat([trainingY, valY]))

print("Mejores hiperparámetros: ", gridSearch.best_params_)

bestM = gridSearch.best_estimator_
predictionYTest = bestM.predict(testingX)

print(
    "Puntaje R2 en set de testing usando el mejor modelo: ",
    r2_score(testingY, predictionYTest),
)