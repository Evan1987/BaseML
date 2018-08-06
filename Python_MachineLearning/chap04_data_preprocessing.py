
# coding: utf-8

import numpy as np
import pandas as pd
from io import StringIO
# 1. Data load

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''
data = pd.read_csv(StringIO(csv_data))

# 2. drop na values
print(data.isnull().sum())
print(data.dropna(axis=0))
print(data.dropna(axis=1))
print(data.dropna(axis=0, how='all'))
print(data.dropna(axis=1,thresh=2))
# 均值插补缺失值
from sklearn.preprocessing import Imputer
imr = Imputer(axis=0, missing_values='NaN', strategy='mean')
imputed_data = imr.fit_transform(data)
print(imputed_data)

