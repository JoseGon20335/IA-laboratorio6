import pandas as pd
import re

print("Loading dataset...")
data = pd.read_csv("CompleteDataset.csv")

def calculate(cell):
    match = re.match(r"(\d+)\s*([+-])\s*(\d+)", str(cell))
    if match:
        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        if op == "+":
            return a + b
        else:
            return a - b
    else:
    
        return cell


data = data.apply(lambda x: x.apply(calculate))
data = data.dropna()
data.drop(["ID", "Name", "Age", "Photo", "Nationality", "Flag", "Club Logo", "Value",
          "Wage", "Special", "Preferred Positions", "Number", "Club", ], axis=1, inplace=True)
data.to_csv("DataCleaned.csv", index=False)
