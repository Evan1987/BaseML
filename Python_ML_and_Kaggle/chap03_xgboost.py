
# coding: utf-8

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

path = "F:/for learn/Python_ML_and_Kaggle/Datasets/"
titanic = pd.read_csv(path + "titanic.txt")


x = titanic[["pclass", "age", "sex"]]
y = titanic["survived"]
x["age"].fillna(x["age"].mean(), inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

vec = DictVectorizer(sparse=False)
vec.fit(x.to_dict(orient="record"))
x_train = vec.transform(x_train.to_dict(orient="record"))
x_test = vec.transform(x_test.to_dict(orient="record"))

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
print("the acc of RF is:", rfc.score(x_test, y_test))

# 采用 xgboost的默认配置进行月预测
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print("the acc of XGBoost is:", xgb.score(x_test, y_test))

