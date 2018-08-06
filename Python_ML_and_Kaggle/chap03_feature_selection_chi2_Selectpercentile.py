
# coding: utf-8


# feature selection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


path = "F:/for learn/Python_ML_and_Kaggle/Datasets/"
titanic = pd.read_csv(path + "titanic.txt")

y = titanic["survived"]
x = titanic.drop(["row.names", "name", "survived"], axis=1)
x["age"].fillna(x["age"].mean(),inplace=True)
x.fillna("Unknown", inplace=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

vec = DictVectorizer()
vec.fit(x.to_dict(orient="record"))
x_train = vec.transform(x_train.to_dict(orient="record"))
x_test = vec.transform(x_test.to_dict(orient="record"))
print(vec.feature_names_)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_predict = dt.predict(x_test)
print("the score of DT without filtering features is ",dt.score(x_test, y_test))


# 引入根据卡方相关性的特征选择器
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
import pylab as pl
# 利用 5折CV法 在训练集上对合适的特征选择量进行验证
percentiles = range(1, 100, 2)
results = []
for i in percentiles:
    fs = SelectPercentile(score_func=chi2, percentile=i)
    x_train_fs = fs.fit_transform(x_train, y_train)
    # 由于是5折验证， 所以输出score时是5个
    scores = cross_val_score(dt, x_train_fs, y_train, cv=5)
    results.append(scores.mean())
results = np.array(list(map(lambda x: round(x, 4), results)))
print(results)
print("the Optimal Number of Features is %d" % (percentiles[results.argmax()]))


pl.plot(percentiles, results)
pl.xlabel("percentile of features")
pl.ylabel("acc")
pl.show()


# 利用得到的最优参数重新训练，并对测试集进行预测
fs = SelectPercentile(score_func=chi2, percentile=percentiles[results.argmax()])
x_train_fs = fs.fit_transform(x_train, y_train)
selectedFeatures = np.array(vec.feature_names_)[fs.get_support()]
dt.fit(x_train_fs, y_train)
x_test_fs = fs.transform(x_test)
print("the score of DT with filtering features is ",dt.score(x_test_fs, y_test))

