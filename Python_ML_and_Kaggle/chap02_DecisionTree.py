
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
# titanic.dtypes


x = titanic[["age", "sex", "pclass"]]
y = titanic["survived"]


x.info()


x["age"].fillna(x["age"].mean(), inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)


# one-Hot
vec = DictVectorizer(sparse=False)
vec.fit(x.to_dict(orient="record"))
x_train = vec.transform(x_train.to_dict(orient="record"))
x_test = vec.transform(x_test.to_dict(orient="record"))
# x_train = pd.DataFrame(x_train,columns=vec.feature_names_)
# x_test = pd.DataFrame(x_test,columns=vec.feature_names_)


dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_predict = dtc.predict(x_test)

print(classification_report(y_true=y_test, y_pred=y_predict,target_names=['died', 'survived']))


print(dtc.score(x_test, y_test))

