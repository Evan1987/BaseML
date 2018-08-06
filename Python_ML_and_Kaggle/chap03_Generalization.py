
# coding: utf-8
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
import pylab as pl


x_train = np.array([6, 8, 10, 14, 18]).reshape((-1, 1))
y_train = np.array([7, 9, 13, 17.5, 18]).reshape((-1, 1))
x_test = np.array([6, 8, 11, 16]).reshape((-1, 1))
y_test = np.array([8, 12, 15, 18]).reshape((-1, 1))
x = np.append(x_train, x_test)
y = np.append(y_train, y_test)

xx = np.linspace(start=0, stop=25, num=101).reshape((-1, 1))
degrees = [1, 2, 4]
cols = ["black", "pink", "green"]
params = list(zip(degrees, cols))
pl.scatter(x, y, c="red")
pl.axis([0, 25, 0, 25])
pl.xlabel("Diameter")
pl.ylabel("Price")

for degree,col in params:
    if degree > 1:
        poly = PolynomialFeatures(degree=degree)
        x_train2 = poly.fit_transform(x_train)
        xx_test = poly.transform(xx)
        x_test2 = poly.transform(x_test)
    else:
        x_train2 = x_train
        x_test2 = x_test
        xx_test = xx
    regressor = LinearRegression()
    regressor.fit(x_train2, y_train)
    fitScore = regressor.score(x_train2, y_train)
    testScore = regressor.score(x_test2, y_test)
    print("The R-squared value of %d-degree training is %f" % (degree, fitScore))
    print("The R-squared value of %d-degree testing is %f" % (degree, testScore))
    yy = regressor.predict(xx_test)
    pl.plt.plot(xx, yy, col, label=("Degree=%d" % degree))
pl.plt.legend()
pl.show()


xx = np.linspace(start=0, stop=25, num=101).reshape((-1, 1))
poly4 = PolynomialFeatures(degree=4)
x_train_poly4 = poly4.fit_transform(x_train)
x_test_poly4 = poly4.transform(x_test)
xx_test = poly4.transform(xx)
methods = ["normal", "Lasso", "Ridge"]
cols = ["green", "blue", "purple"]
params = list(zip(methods, cols))
pl.scatter(x, y, c="red")
pl.axis([0, 25, 0, 25])
pl.xlabel("Diameter")
pl.ylabel("Price")

for method, col in params:
    obj = LinearRegression() if method == "normal" else eval(method + "()")
    obj.fit(x_train_poly4, y_train)
    fitScore = obj.score(x_train_poly4, y_train)
    testScore = obj.score(x_test_poly4, y_test)
    yy = obj.predict(xx_test)
    print(method)
    print("The R-squared value of %s regression training is %f" % (method, fitScore))
    print("The R-squared value of %s regression testing is %f" % (method, testScore))
    print(obj.coef_)
    print("****************************************")
    pl.plt.plot(xx, yy, col, label=method)
pl.plt.legend()
pl.show()

