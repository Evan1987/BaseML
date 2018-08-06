
# coding: utf-8
import numpy as np
import pandas as pd
def dataInit():
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']
    ])
    df.columns = ['color', 'size', 'price', 'classLabel']
    return df


df = dataInit()

# 1.有序类别特征映射
size_mapping = {'M': 1, 'L': 2, 'XL': 3}
df['size'] = df['size'].map(size_mapping)
df.head()
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
df.head()

# 2.无序类别特征映射
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classLabel']))}
df['classLabel'] = df['classLabel'].map(class_mapping)
df.head()
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classLabel'] = df['classLabel'].map(inv_class_mapping)
df.head()

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df['classLabel'] = enc.fit_transform(df['classLabel'].values)
df.head()
df['classLabel'] = enc.inverse_transform(df['classLabel'].values)
df.head()

# 3.独热编码
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
cols = ['color', 'size', 'classLabel']
lenc = LabelEncoder()
for col in cols:
    df[col] = lenc.fit_transform(df[col])

ohenc = OneHotEncoder(categorical_features=[0, 3])
df = ohenc.fit_transform(df.values)
print(df.toarray())

# 4. 特征归一化
from sklearn.preprocessing import MinMaxScaler
df = dataInit()
mms = MinMaxScaler()
df['price'] = mms.fit_transform(df['price'].values.reshape(-1, 1))
df.head()

# 5. 特征标准化
from sklearn.preprocessing import StandardScaler
df = dataInit()
scaler = StandardScaler()
df['price'] = scaler.fit_transform(df['price'].values.reshape(-1, 1))
df.head()

