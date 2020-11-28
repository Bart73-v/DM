import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re


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


