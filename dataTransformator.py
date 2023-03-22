import pandas as pd
import re

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("CompleteDataset.csv")


def calculate(cell):
    match = re.match(r"(\d+)\s*([+-])\s*(\d+)", str(cell))
    if match:
        # If the cell contains a valid operation, calculate and return the result
        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        if op == "+":
            return a + b
        else:
            return a - b
    else:
        # If the cell does not contain a valid operation, return the original value
        return cell


# Apply the calculate function to each element of the dataset
data = data.apply(lambda x: x.apply(calculate))

data = data.dropna()

data.drop(["ID", "Name", "Age", "Photo", "Nationality", "Flag", "Club Logo", "Value",
          "Wage", "Special", "Preferred Positions", "Number", "Club", ], axis=1, inplace=True)

data.to_csv("CompleteDatasetTransformed.csv", index=False)
