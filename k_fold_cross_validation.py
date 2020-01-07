import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tree import DecisionTreeClassifier
from id3 import Id3Estimator
from sklearn.preprocessing import OneHotEncoder

K_VAL = 5
MAX_DEPTH = 3

if __name__ == '__main__':
    df = pd.read_csv("data/sampled_data.csv")
    length = df.shape[0]
    k_valid = [int((k*length)/K_VAL) for k in range(K_VAL+1)]
    id3TN, id3FP, id3FN, id3TP = 0, 0, 0, 0
    for k in range(K_VAL):
        id3 = DecisionTreeClassifier()
        X = df.iloc[(df.index < k_valid[k]) | (df.index >= k_valid[k+1]), :-1]
        Y = df.iloc[(df.index < k_valid[k]) | (df.index >= k_valid[k+1]), -1]
        id3.fit(X, Y, max_depth=MAX_DEPTH)
        pred = id3.predict(df.iloc[k_valid[k]:k_valid[k+1], :-1])
        cm = confusion_matrix(df.iloc[k_valid[k]:k_valid[k+1], -1], pred)
        tn, fp, fn, tp = cm.ravel()
        id3TN += tn
        id3FP += fp
        id3FN += fn
        id3TP += tp
        # cmx = sns.heatmap(cm, annot=True, fmt='g')
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.show()
    categories = list(df.columns[:-1])
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[categories])
    train_ohe = pd.DataFrame(encoded, columns=np.hstack(encoder.categories_))
    train_df = pd.concat((df.iloc[:,:-1], train_ohe), axis=1).drop(categories, axis=1)
    estTN, estFP, estFN, estTP = 0, 0, 0, 0
    for k in range(K_VAL):
        estimator = Id3Estimator(max_depth=MAX_DEPTH-1)
        X = train_df.iloc[(train_df.index < k_valid[k]) | (train_df.index >= k_valid[k+1])]
        Y = df.iloc[(df.index < k_valid[k]) | (df.index >= k_valid[k+1]), -1]
        estimator.fit(X, Y)
        pred = estimator.predict(train_df.values[k_valid[k]:k_valid[k+1]])
        cm = confusion_matrix(df.iloc[k_valid[k]:k_valid[k+1], -1], pred)
        tn, fp, fn, tp = cm.ravel()
        estTN += tn
        estFP += fp
        estFN += fn
        estTP += tp
        # cmx = sns.heatmap(cm, annot=True, fmt='g')
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.show()
    print("K_VAL="+str(K_VAL)+" MAX_DEPTH="+str(MAX_DEPTH)+"\nCM|our|est\nTN|"+str(id3TN)+","+str(estTN)+"\nFP|"+str(id3FP)+","+str(estFP)+"\nFN|"+str(id3FN)+","+str(estFN)+"\nTP|"+str(id3TP)+","+str(estTP))
