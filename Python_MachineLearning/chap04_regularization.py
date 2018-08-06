
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df_wine.columns = ['ClassLabel',
                   'Alcohol', 
                   'Malic acid', 
                   'Ash', 
                   'Alcalinity of ash', 
                   'Magnesium', 
                   'Total phenols', 
                   'Flavanoids', 
                   'Nonflavanoid phenols', 
                   'Proanthocyanins', 
                   'Color intensity', 
                   'Hue', 
                   'OD280/OD315 of diluted wines', 
                   'Proline']

X = df_wine[df_wine.columns[1:]]
y = df_wine[df_wine.columns[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(C=0.1, penalty='l1')
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
print("the train ACC of lr is %.4f" % lr.score(X_train, y_train))
print("the test ACC of lr is %.4f" % lr.score(X_test, y_test))
print(lr.coef_)

fig = plt.figure(1)
ax = plt.subplot(111)
colors = ['blue', 
          'green', 
          'red', 
          'cyan', 
          'magenta', 
          'yellow', 
          'black', 
          'pink', 
          'lightgreen', 
          'lightblue', 
          'gray', 
          'indigo', 
          'orange']
weights, params = [], []
# 在不同惩罚系数下，各特征的权重系数
for c in np.arange(-4, 6, dtype=float):
    lr = LogisticRegression(C=(10**c), penalty='l1', random_state=0)
    lr.fit(X_train, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.mat(weights)

for idx, color in enumerate(colors):
    plt.plot(params, weights[:, idx].A1, label=df_wine.columns[idx+1], color=color)
plt.axhline(y=0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**-5, 10**5])
plt.ylabel('weight coef')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()

