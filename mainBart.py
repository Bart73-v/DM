import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

df = pd.read_csv('.\student-mat.csv', sep=';')
data = np.array(df)
attributes = np.array(df.columns)

print(attributes)
