
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

path = "F:/for learn/Python_MachineLearning/"
df = pd.read_csv(path + "wdbc.data", header=None)

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipe_lr = Pipeline([("scaler", StandardScaler()), 
                    ('pca', PCA(n_components=2)),
                    ('lr', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print("the Test ACC is: %.3f" % pipe_lr.score(X_test, y_test))


kfold = StratifiedKFold(n_splits=10, random_state=1)

scores = []
for k, (train, test) in enumerate(kfold.split(X_train, y_train)):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print("Fold: %d, Class dist: %s, ACC: %.3f" % (k+1, np.bincount(y_train[train]), score))

print("CV ACC: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

# 采用sklearn中的集成方法做cv
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
print("CV ACC scores: %s" % scores)
print("CV ACC: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

