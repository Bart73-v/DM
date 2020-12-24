import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold

#-----------------------------------------------------------

def getData(mode):
    df = pd.read_csv(R'.\student-mat.csv', sep=';')
    dy = np.array(df['G3'])
    if mode==1:
        df = df.drop(columns=["G1", "G2", "G3"])
    elif mode == 2:
        df = df.drop(columns="G3")
    df = encode(df)
    df.head()
    data = np.array(df)
    dAttributes = np.array(df.columns)
    return data, dy, dAttributes


def encode(datframe):
        # encode binary nominal attributes
    cleanup_bins = {"school" :      {"GP" : -1, "MS" : 1},
                    "sex" :         {"F" : -1, "M" : 1},
                    "address" :     {"U" : -1, "R" : 1},
                    "famsize" :     {"LE3": -1, "GT3" : 1},
                    "Pstatus" :     {"T" : -1, "A" : 1}}
    datframe = datframe.replace(cleanup_bins)
        # encode other nominal attributes using dummy variables
    one_hot_MF = pd.get_dummies(datframe["Mjob"]).join(pd.get_dummies(datframe["Fjob"]), lsuffix='_M', rsuffix='_F')
    one_hot_RG = pd.get_dummies(datframe["reason"]).join(pd.get_dummies(datframe["guardian"]), lsuffix='_R', rsuffix='_G')
    datframe = datframe.drop(columns=["Mjob", "Fjob", "reason", "guardian"])
    datframe = datframe.join([one_hot_MF, one_hot_RG])
    datframe = datframe.replace({'yes': 1, 'no': 0})
    return datframe
    

def createYBin(y):
    result = np.copy(y)
    for ind, val in enumerate(result):
        if val >= 10:
            result[ind] = 1
        else: result[ind] = 0
    return result


def createYClas(y):
    result = np.copy(y)
    for ind, val in enumerate(result):
        if val >= 16:
            result[ind] = 4
        elif val >= 12:
            result[ind] = 3
        elif val >= 8:
            result[ind] = 2
        elif val >= 4:
            result[ind] = 1
        else: result[ind] = 0
    return result


def clf_tree(samples, target, attributes, i): #https://scikit-learn.org/stable/modules/tree.html#classification
    clf = tree.DecisionTreeClassifier(max_depth=i) #max_depth=
    #clf.min_samples_split = 10
    kf = KFold(n_splits=10)
    accuracy = []
    for train_index, test_index in kf.split(samples):
        X_train, X_test = samples[train_index], samples[test_index]
        y_train, y_test = target[train_index], target[test_index]
        fold_acc = []
        for i in range(0,5):
            clf.fit(X_train, y_train.ravel())
            fold_acc.append(clf.score(X_test, y_test))
        accuracy.append(max(fold_acc))
    return np.mean(accuracy)

#-----------------------------------------------------------


X, y, attributes = getData(2)
yBin = createYBin(y)
depth = []
for i in range(1,100):
    depth.append([clf_tree(X,yBin,attributes,i), i])
print(max(depth))

