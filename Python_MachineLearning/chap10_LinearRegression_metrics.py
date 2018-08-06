
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


path = "F:/for learn/Python_MachineLearning/"
df = pd.read_csv(path + "housing.data", 
                 header=None, 
                 sep="\s+")
df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]


X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


plt.scatter(y_train_pred, y_train_pred-y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred-y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

print('MSE of train is: %.3f' % mean_squared_error(y_pred=y_train_pred, y_true=y_train))
print('MSE of test is: %.3f' % mean_squared_error(y_pred=y_test_pred, y_true=y_test))

print('R2 of train is: %.3f' % r2_score(y_pred=y_train_pred, y_true=y_train))
print('R2 of test is: %.3f' % r2_score(y_pred=y_test_pred, y_true=y_test))


# 正则方法：
from sklearn.linear_model import Lasso, Ridge, ElasticNet
tp = ['train', 'test']
method = {"mse": mean_squared_error, "r2": r2_score}
# 组合多个命令
command = np.stack(np.meshgrid(tp, list(method.keys())), axis=-1).reshape(-1, 2)
for model in [Ridge(alpha=1.0), Lasso(alpha=1.0), ElasticNet(alpha=1.0, l1_ratio=0.5)]:
    modelName = model.__class__.__name__
    model.fit(X_train, y_train)
    for tp, metric in command:
        y_pred = model.predict(eval("X_" + tp))
        y_true = eval("y_" + tp)
        score = method[metric](y_pred=y_pred, y_true=y_true)
        print('%s of %s by %s is: %.3f ' % (metric, tp, modelName, score))

