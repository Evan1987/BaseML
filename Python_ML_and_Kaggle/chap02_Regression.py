
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Data prepared
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
x = data.copy(deep=True)
y = data["y"] = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
ss_x = StandardScaler()
ss_y = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
# reshape make y to a column-like array
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))
ori_y_test = ss_y.inverse_transform(y_test)

# Linear regression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)

# SGD regression
sgdr = SGDRegressor()
sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)

for model, result in [(lr, lr_y_predict), (sgdr, sgdr_y_predict)]:
    modelName = model.__class__.__name__
    ori_result = ss_y.inverse_transform(result)
    # default score is precisely the r2_score
    modelScore = model.score(x_test, y_test)
    mse = mean_squared_error(y_true=ori_y_test, y_pred=ori_result)
    mae = mean_absolute_error(y_true=ori_y_test, y_pred=ori_result)
    r2 = r2_score(y_true=ori_y_test, y_pred=ori_result)
    print("the default measure score of %s is %f" % (modelName, modelScore))
    print("the mae of %s is %f" % (modelName, mae))
    print("the mse of %s is %f" % (modelName, mse))
    print("the r2_score of %s is %f" % (modelName, r2))
    print("\n")


# SVR regression
# ****实践了一些反射技巧
for kernel in ['linear', 'poly', 'rbf']:
    var = kernel + "_svr"
    locals()[var] = SVR(kernel=kernel)
    obj = eval(var)
    obj.fit(x_train, y_train)
    result = obj.predict(x_test)
    ori_result = ss_y.inverse_transform(result)
    r2 = r2_score(y_true=ori_y_test, y_pred=ori_result)
    #r2 score doesn't shift wherever result is standarized back.
    #r2_s = r2_score(y_true=y_test, y_pred=result)
    mae = mean_absolute_error(y_true=ori_y_test, y_pred=ori_result)
    mse = mean_squared_error(y_true=ori_y_test, y_pred=ori_result)
    print("R2 value of kernel %s is %f" % (kernel, r2))
    #print("R2-s value of kernel %s is %f" % (kernel, r2_s))
    print("mae value of kernel %s is %f" % (kernel, mae))
    print("mse value of kernel %s is %f" % (kernel, mse))
    print("\n")



# KNN regression 
## 平均回归 和 按距离加权回归
for weights in ["uniform", "distance"]:
    knr = KNeighborsRegressor(weights=weights)
    knr.fit(x_train, y_train)
    result = knr.predict(x_test)
    ori_result = ss_y.inverse_transform(result)
    r2 = r2_score(y_true=ori_y_test, y_pred=ori_result) 
    mae = mean_absolute_error(y_true=ori_y_test, y_pred=ori_result)
    mse = mean_squared_error(y_true=ori_y_test, y_pred=ori_result)
    print("R2 value of weights %s is %f" % (weights, r2))
    print("mae value of weights %s is %f" % (weights, mae))
    print("mse value of weights %s is %f" % (weights, mse))
    print("\n")


# Tree regression
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
result = dtr.predict(x_test)
ori_result = ss_y.inverse_transform(result)
r2 = r2_score(y_true=ori_y_test, y_pred=ori_result) 
mae = mean_absolute_error(y_true=ori_y_test, y_pred=ori_result)
mse = mean_squared_error(y_true=ori_y_test, y_pred=ori_result)
print("R2 value of DecisionTreeRegressor is %f" % r2)
print("mae value of DecisionTreeRegressor is %f" % mae)
print("mse value of DecisionTreeregressor is %f" % mse)


# Ensemble regression
ensembleList = [RandomForestRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor()]
for model in ensembleList:
    modelName = model.__class__.__name__
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    ori_result = ss_y.inverse_transform(result)
    r2 = r2_score(y_true=ori_y_test, y_pred=ori_result) 
    mae = mean_absolute_error(y_true=ori_y_test, y_pred=ori_result)
    mse = mean_squared_error(y_true=ori_y_test, y_pred=ori_result)
    for metric in ["r2", "mae", "mse"]:
        print("%s value of %s is %f" % (metric.upper(), modelName, eval(metric)))
    feature_importance = list(zip(model.feature_importances_, x.columns))
    print(np.sort(feature_importance, axis=0))
    print("\n")

