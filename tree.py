import random
import math


class tree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def ent(self, data):
        temp = {}
        for row in data:
            label = row[-1]
            if label not in temp:
                temp[label] = 0
            temp[label] += 1
        ent = 0
        for label in temp:
            probability = temp[label] / float(len(data))
            ent -= probability * math.log(probability, 2)
        return ent

    def splitData(self, data, column, value):
        left = []
        right = []
        for row in data:
            if row[column] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def bestSplit(self, data):
        bestEnt = float('inf')
        bestCol = None
        bestVal = None
        for column in range(len(data[0])-1):
            values = set([row[column] for row in data])
            for value in values:
                left, right = self.splitData(data, column, value)
                if len(left) == 0 or len(right) == 0:
                    continue
                ent = (len(left) / len(data)) * self.ent(left) + \
                    (len(right) / len(data)) * self.ent(right)
                if ent < bestEnt:
                    bestEnt = ent
                    bestCol = column
                    bestVal = value
        return bestCol, bestVal



    def buildTree(self, data, depth=0):
        if depth >= self.max_depth:
            return max(set([row[-1] for row in data]), key=[row[-1] for row in data].count)
        print('hola')
        if len(set([row[-1] for row in data])) == 1:
            return data[0][-1]
        column, value = self.bestSplit(data)
        left, right = self.splitData(data, column, value)
        node = {'column': column, 'value': value}
        node['left'] = self.buildTree(left, depth+1)
        node['right'] = self.buildTree(right, depth+1)
        return node

    def predict(self, row, tree):
        if isinstance(tree, str):
            return tree
        if row[tree['column']] < tree['value']:
            return self.predict(row, tree['left'])
        else:
            return self.predict(row, tree['right'])
