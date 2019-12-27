import copy
import numpy as np
import pandas as pd


class Node:
    def __init__(self, R, S):
        self.children = {}
        self.R = R  # Remaining attributes
        self.S = S  # Indices of data points
        self.decidingAttribute = None
        self.leafClass = None


class DecisionTreeClassifier:
    def __init__(self):
        self.root = None
        self.X = None
        self.Y = None

    def fit(self, X, Y):
        self.X = X.reset_index(drop=True)
        self.Y = Y.reset_index(drop=True)
        self.root = Node(X.columns, [i for i in range(0, len(X))])
        self.__id3__(self.root)

    def __id3__(self, node):
        if len(node.S) == 0:
            raise Exception('Attempt to create a node for no datapoints')
        pointsX = self.X.iloc[node.S]
        pointsY = self.Y.iloc[node.S]
        if len(pointsY.unique()) == 1:
            node.leafClass = pointsY.iloc[0]
            return
        if len(node.R) == 0:
            node.leafClass = pointsY.mode().iloc[0]
            return
        maxGain = ('', -1000)
        for attribute in node.R:
            infGain = self.__infGain__(pointsX[attribute], pointsY)
            if infGain > maxGain[1]:
                maxGain = (attribute, infGain)
        node.decidingAttribute = maxGain[0]
        attributeValues = pointsX[node.decidingAttribute].unique()
        for attributeVal in attributeValues:
            attrValuePointsX = pointsX.loc[pointsX[node.decidingAttribute] == attributeVal]
            node.children[attributeVal] = Node([n for n in node.R if n != node.decidingAttribute],
                                               copy.deepcopy(attrValuePointsX.index))
            self.__id3__(node.children[attributeVal])

    @staticmethod
    def __infGain__(X, Y):
        freqY = Y.value_counts(normalize=True).to_numpy()
        entropy = (-1) * np.dot(freqY, np.log(freqY))
        subsetEntropy = 0
        attributeValues = X.unique()
        for attributeVal in attributeValues:
            subsetX = X.loc[X == attributeVal]
            subsetY = Y.loc[X == attributeVal]
            freqY = subsetY.value_counts(normalize=True).to_numpy()
            subsetEntropy += float(len(subsetX)) / len(X) * (-1) * np.dot(freqY, np.log(freqY))
        return entropy - subsetEntropy

    def predict(self, X):
        predictions = []
        for index, x in X.iterrows():
            node = self.root
            while 1:
                if node.leafClass is not None:
                    predictions.append(node.leafClass)
                    break
                node = node.children[x.loc[node.decidingAttribute]]
        return pd.Series(predictions)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        Y = Y.reset_index(drop=True)
        cmp = Y.eq(predictions)
        accuracy = float(len(cmp.loc[cmp == 1])) / len(Y)
        print(accuracy)
        return accuracy


if __name__ == '__main__':
    data = pd.read_csv("data.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    model = DecisionTreeClassifier()
    model.fit(data.iloc[:2000, :-1], data.iloc[:2000, -1])
    model.evaluate(data.iloc[2000:, :-1], data.iloc[2000:, -1])
