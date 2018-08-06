
# coding: utf-8

# In[19]:

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


news = fetch_20newsgroups(subset="all")
data = news.data[:3000]
label = news.target[:3000]

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=33)

stages = [("vec", TfidfVectorizer(stop_words="english", analyzer="word")), 
          ("svc", SVC())]
clf = Pipeline(stages)

# Gamma： rbf核函数的参数
# C： 软间隔参数
# 一共是12个不同组合
parameters = {"svc__gamma": np.logspace(-2, 1, 4), 
              "svc__C": np.logspace(-1, 1, 3)}

# 利用 pipeline1进行 参数网格搜索
# refit = True 将最佳参数应用于训练集和验证集
# cv=3 3折CV检验（一共进行 36次循环）
# verbose=2 控制信息输出，越大，输出信息越多
# n_job： 并行搜索， -1为用所有的核心
gs = GridSearchCV(clf, param_grid=parameters, verbose=2, refit=True, cv=3, n_jobs=-1)

# get_ipython().run_line_magic('time', '_ = gs.fit(x_train, y_train)')

print(gs.best_params_)
print(gs.best_score_)
print(gs.score(x_test, y_test))

bestModel = gs.best_estimator_
bestModel.score(x_test, y_test)

