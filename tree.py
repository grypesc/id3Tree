import copy
import numpy as np
import pandas as pd


class Node:
    def __init__(self, R, S, depth):
        self.children = {}
        self.R = R  # Remaining attributes
        self.S = S  # Indices of data points
        self.deciding_attribute = None
        self.leaf_class = None
        self.depth = depth


class DecisionTreeClassifier:
    def __init__(self):
        self.root = None
        self.X = None
        self.Y = None
        self.max_depth = None

    def fit(self, X, Y, max_depth=-1):
        self.X = X.reset_index(drop=True)
        self.Y = Y.reset_index(drop=True)
        self.max_depth = max_depth
        if max_depth == -1:
            self.max_depth = len(X.columns) + 1
        self.root = Node(X.columns, [i for i in range(0, len(X))], 1)
        self.__id3__(self.root)

    def __id3__(self, node):
        if len(node.S) == 0:
            raise Exception('Attempt to create a node for no datapoints')
        points_X = self.X.iloc[node.S]
        points_Y = self.Y.iloc[node.S]
        if len(points_Y.unique()) == 1:
            node.leaf_class = points_Y.iloc[0]
            return
        if len(node.R) == 0 or node.depth >= self.max_depth:
            node.leaf_class = points_Y.mode().iloc[0]
            return
        max_gain = ('', -1000)
        for attribute in node.R:
            inf_gain = self.__infGain__(points_X[attribute], points_Y)
            if inf_gain > max_gain[1]:
                max_gain = (attribute, inf_gain)
        node.deciding_attribute = max_gain[0]
        attribute_values = points_X[node.deciding_attribute].unique()
        for attribute_val in attribute_values:
            attr_value_points_X = points_X.loc[points_X[node.deciding_attribute] == attribute_val]
            node.children[attribute_val] = Node([n for n in node.R if n != node.deciding_attribute], copy.deepcopy(attr_value_points_X.index), node.depth + 1)
            self.__id3__(node.children[attribute_val])

    @staticmethod
    def __infGain__(X, Y):
        freq_Y = Y.value_counts(normalize=True).to_numpy()
        entropy = (-1) * np.dot(freq_Y, np.log(freq_Y))
        subset_entropy = 0
        attribute_values = X.unique()
        for attribute_val in attribute_values:
            subset_X = X.loc[X == attribute_val]
            subset_Y = Y.loc[X == attribute_val]
            freq_Y = subset_Y.value_counts(normalize=True).to_numpy()
            subset_entropy += float(len(subset_X)) / len(X) * (-1) * np.dot(freq_Y, np.log(freq_Y))
        return entropy - subset_entropy

    def predict(self, X):
        predictions = []
        for index, x in X.iterrows():
            node = self.root
            while 1:
                if node.leaf_class is not None:
                    predictions.append(node.leaf_class)
                    break
                if x.loc[node.deciding_attribute] not in node.children:  # Faced attribute value that wasn't present in training set
                    points_Y = self.Y.iloc[node.S]
                    predictions.append(points_Y.mode().iloc[0])
                    break
                node = node.children[x.loc[node.deciding_attribute]]
        return pd.Series(predictions)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        Y = Y.reset_index(drop=True)
        cmp = Y.eq(predictions)
        accuracy = float(len(cmp.loc[cmp == 1])) / len(Y)
        return accuracy


if __name__ == '__main__':
    data = pd.read_csv("data.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    model = DecisionTreeClassifier()
    model.fit(data.iloc[:3000, :-1], data.iloc[:3000, -1], max_depth=-1)
    acc = model.evaluate(data.iloc[3000:, :-1], data.iloc[3000:, -1])
    print(acc)
