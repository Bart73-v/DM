import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn import tree
import re


# Mean-center & normalize data
# Alter nominal data
# Hyper parameters?
# 

#-----------------------------------------------------------

def encode(df):
    encoder = LabelEncoder()
    for column in df.columns:
        for unique in df[column].unique():
            if isinstance(unique, str):
                if not re.match('^[0-9]*$', unique):
                    df[column] = encoder.fit_transform(df[column])
                    break
    return df


def createYBin(y):
    for ind, val in enumerate(y):
        if val >= 10:
            y[ind] = 1
        else: y[ind] = 0
    return y


def createYClas(y):
    for ind, val in enumerate(y):
        if val >= 16:
            y[ind] = 4
        elif val >= 12:
            y[ind] = 3
        elif val >= 8:
            y[ind] = 2
        elif val >= 4:
            y[ind] = 1
        else: y[ind] = 0
    return y


def getData():
    df = pd.read_csv('.\student-mat.csv', sep=';')
    df = encode(df)
    data = np.array(df)
    attributes = np.array(df.columns)
    y = data[:,-1]
    data = np.delete(data, -1, 1)
    data = np.delete(data, -1, 1)
    data = np.delete(data, -1, 1)
    return data, y, attributes


def get_clf_tree(max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.min_samples_split = 100
    return clf


def print_clf_tree(X, y, attributes):
    fig = plt.figure(figsize=[20,10])
    clf = get_clf_tree(999)
    clf = clf.fit(X,y)
    fig = tree.plot_tree(clf, feature_names=attributes, fontsize=7)
    plt.show()
    print("Figure 1: ")

#-----------------------------------------------------------

X, y, attributes = getData()
yBin = createYBin(y)
yClas = createYClas(y)

print(X)