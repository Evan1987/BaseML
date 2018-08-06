
# coding: utf-8

# feature extraction
from sklearn.feature_extraction import DictVectorizer

measurements = [{"city": "Dubai", "temperature": 33},
                {"city": "London", "temperature": 12}, 
                {"city": "San Fransisco", "temperature": 18}]

vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news = fetch_20newsgroups(subset="all")
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)


import itertools as it
# 在两种词频向量特征中，采用是否去除停用词的不同参数进行预测
objStr = ["CountVectorizer", "TfidfVectorizer"]
stopWordsParam = [None, "'english'"]
methods = it.product(objStr, stopWordsParam)

for objStr, stopWordsParam in methods:
    paraName = "without" if stopWordsParam == None else "with"
    methodName = objStr + " " + paraName + " " + "filtering Stopwords"
    obj = eval(objStr + "(stop_words=%s)" % stopWordsParam)
    obj_x_train = obj.fit_transform(x_train)
    obj_x_test = obj.transform(x_test)
    mnb = MultinomialNB()
    mnb.fit(obj_x_train, y_train)
    score = mnb.score(obj_x_test, y_test)
    y_predict = mnb.predict(obj_x_test)
    print("the ACC score of %s is: %f" % (methodName, score))
    print(classification_report(y_true=y_test, y_pred=y_predict, target_names=news.target_names))
