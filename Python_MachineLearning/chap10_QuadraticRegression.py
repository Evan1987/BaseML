
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

X = np.array([258.0, 270.0 ,294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0]).reshape(-1, 1)
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])
x_fit = np.arange(250, 600, 10).reshape(-1, 1)

lr = LinearRegression()
pr = LinearRegression()

# 处理后的特征具有 degree+1列：X^0, X^1, X^2
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)
x_quad_fit = quadratic.fit_transform(x_fit)

lr.fit(X, y)
pr.fit(X_quad, y)

y_lin_fit = lr.predict(x_fit)
y_quad_fit = pr.predict(x_quad_fit)

plt.scatter(X, y, label='training points')
plt.plot(x_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(x_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')
plt.show()


y_lr_pred = lr.predict(X)
y_pr_pred = pr.predict(X_quad)
print("the mse of Linear Model is %.3f" % mean_squared_error(y_pred=y_lr_pred, y_true=y))
print("the mse of Quadratic Model is %.3f" % mean_squared_error(y_pred=y_pr_pred, y_true=y))

print("the r2 of Linear Model is %.3f" % r2_score(y_pred=y_lr_pred, y_true=y))
print("the r2 of Quadratic Model is %.3f" % r2_score(y_pred=y_pr_pred, y_true=y))

