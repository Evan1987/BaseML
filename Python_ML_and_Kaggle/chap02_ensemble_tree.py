
# coding: utf-8

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

titanic = pd.read_csv("F:/for learn/Python_ML_and_Kaggle/Datasets/titanic.txt")

x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
x['age'].fillna(x.age.mean(), inplace=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)


vec = DictVectorizer(sparse=False)
vec.fit(x.to_dict(orient="record"))
x_train = vec.transform(x_train.to_dict(orient="record"))
x_test = vec.transform(x_test.to_dict(orient="record"))


dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_y_predict = dtc.predict(x_test)


rfc = RandomForestClassifier(max_depth=3,n_estimators=10)
rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)


gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_predict = gbc.predict(x_test)


for model, result in [(dtc, dtc_y_predict), (rfc, rfc_y_predict), (gbc, gbc_y_predict)]:
    print("the ACC of %s is %f" % (model.__class__.__name__, model.score(x_test, y_test)))
    print(classification_report(y_true=y_test, y_pred=result))

