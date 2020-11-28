import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn import tree
import re

def encode(df):
    encoder = LabelEncoder()
    for column in df.columns:
        for unique in df[column].unique():
            if isinstance(unique, str):
                if not re.match('^[0-9]*$', unique):
                    df[column] = encoder.fit_transform(df[column])
                    break
    return df

df = pd.read_csv('.\student-mat.csv', sep=';')
df = encode(df)

data = np.array(df)
attributes = np.array(df.columns)
y = data[:,-1]
for ind, grade in enumerate(y):
    if grade >= 10:
        y[ind] = 1
    else: y[ind] = 0


data = np.delete(data, -1, 1)
data = np.delete(data, -1, 1)
data = np.delete(data, -1, 1)



def get_clf_tree(max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.min_samples_split = 100
    return clf


fig = plt.figure(figsize=[20,10])
clf = get_clf_tree(999)
clf = clf.fit(data,y)
fig = tree.plot_tree(clf, feature_names=attributes, fontsize=7)
plt.show()
print("Figure 1: Decision tree for the Wine dataset with a sample split of 100")
