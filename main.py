import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# def ReadFile(path):
#     df = pd.read_csv(path, sep=';')
#     df, indexDict = Encode(df)
#     X = np.array(df.iloc[:,:-3])
#     y = df.iloc[:,-1]
#     y = np.array([0 if x < 10 else 1 for x in y])
#     return X, y

# def Encode(df):
#     indexDict = dict()
#     encoder = LabelEncoder()
#     for column in df.columns:
#         for unique in df[column].unique():
#             if isinstance(unique, str):
#                 if not re.match('^[0-9]*$', unique):
#                     uniques = df[column].unique()
#                     df[column] = encoder.fit_transform(df[column])
#                     encoding = df[column].unique()

#                     indexDict[column] = dict()
#                     for i in range(len(uniques)):
#                         indexDict[column][encoding[i]] = uniques[i]
#                     break
#     return df, indexDict

def ReadFile(path):
    df = pd.read_csv(path, sep=';')
    y = df.iloc[:,-1]
    y = np.array([0 if x < 10 else 1 for x in y])

    X = np.array(Encode(df.iloc[:,:-3]))
    print(X.shape)
    return X, y

def Encode(df):
    return OneHotEncoder().fit_transform(df).toarray()
    

def Regression(path):
    X, y = ReadFile(path)
    kf = KFold(n_splits=5, random_state=10, shuffle=True)
    model = LogisticRegression(solver='liblinear')
    result = cross_val_score(model, X, y, cv = kf)
    print("Avg regression accuracy: {}".format(result.mean()))

def SVM(path):
    X, y = ReadFile(path)
    # model = SVC(kernel = 'linear')
    # # model = SVC(kernel = 'linear',gamma = 'scale', shrinking = False)
    # model.fit(X, y)
    kf = KFold(n_splits=5, random_state=10, shuffle=True)
    result = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]    

        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        # kf = KFold(n_splits=5, random_state=10, shuffle=True)
        model = SVC(kernel = 'linear',gamma = 'scale', shrinking = False)
        model.fit(X_train, y_train)
        # result = cross_val_score(model, X, y, cv = kf)
        result.append(model.score(X_test, y_test))

    print("Avg SVM accuracy: {}".format(sum(result)/len(result)))

path = R'C:\Users\anton\Documents\Data science\Data mining\Project\data\student-mat.csv'
SVM(path)
# Regression(path)

