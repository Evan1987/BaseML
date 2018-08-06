
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score


path = "F:/for learn/Python_ML_and_Kaggle/Datasets/Titanic/"


train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

# 0. Data Preparation
## 0.1 initial features 
selected_features = ["Sex", "Pclass", "Age", "Embarked", "SibSp", "Parch", "Fare"]
x_train = train[selected_features]
x_test = test[selected_features]
y_train = train["Survived"]

## 0.2 NA fill
def fillNA(df, na_features, fill_values):
    zipList = list(zip(na_features, fill_values))
    for feature, value in zipList:
        df[feature].fillna(value, inplace=True)
    return df

fillna_Embarked = x_train["Embarked"].value_counts().argmax()
fillna_Ages = x_train["Age"].mean()
fillna_Fare = x_train["Fare"].mean()
na_features = ["Embarked", "Age", "Fare"]
fill_values = [fillna_Embarked, fillna_Ages, fillna_Fare]

x_train = fillNA(x_train, na_features, fill_values)
x_test = fillNA(x_test, na_features, fill_values)

## 0.3 feature ETL
vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient="record"))
x_test = vec.transform(x_test.to_dict(orient="record"))

# 1. Train
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)

xgb = XGBClassifier()
xgb.fit(x_train, y_train)
xgb_y_predict = xgb.predict(x_test)

print(cross_val_score(rfc, x_train, y_train, cv=5).mean())
print(cross_val_score(xgb, x_train, y_train, cv=5).mean())

## 1.1 Single Model with default params
def subMission(y_predict, fileName):
    df = pd.DataFrame({"PassengerID": test["PassengerId"], "Survived": y_predict})
    df.to_csv(fileName, index=False)
subMission(rfc_y_predict, path + "rfc_sub.csv")
subMission(xgb_y_predict, path + "xgb_sub.csv")

## 1.2 Single Model with best params
params = {'max_depth': [2, 3, 4, 5, 6], 
          'n_estimators': [100, 300, 500, 700, 900], 
          'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]
         }
xgb_best = XGBClassifier()
gs = GridSearchCV(estimator=xgb_best, param_grid=params, n_jobs=-1, cv=5, verbose=1)
gs.fit(x_train, y_train)

print(gs.best_params_)

xgb_best_y_predict = gs.predict(x_test)
subMission(xgb_best_y_predict, path + "xgb_best_sub.csv")

