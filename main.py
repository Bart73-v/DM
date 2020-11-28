import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re
from sklearn import tree



def Encode(df):
    encoder = LabelEncoder()
    for column in df.columns:
        for unique in df[column].unique():
            if isinstance(unique, str):
                if not re.match('^[0-9]*$', unique):
                    df[column] = encoder.fit_transform(df[column])
                    break
    return df

df = pd.read_csv(R'C:\Users\anton\Documents\Data science\Data_mining\Project\data\student-mat.csv', sep=';')
df = Encode(df)
print(df)



def get_clf_tree(max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.min_samples_split = 100
    return clf

fig = plt.figure(figsize=[100,100])
clf = tree.DecisionTreeClassifier(max_depth=999)
clf.min_samples_split = 100
clf = clf.fit(X,y)
fig = tree.plot_tree(clf, feature_names=attribute_names, fontsize=40)
plt.show()

