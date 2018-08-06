
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
data = iris.data
labels = iris.target

x_train, x_test, y_train, y_test = train_test_split(data, 
                                                    labels, 
                                                    test_size=0.25, 
                                                    random_state=33)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_predict = knc.predict(x_test)


print("the ACC of the KNN is %f" % knc.score(x_test, y_test))

print(classification_report(y_true=y_test, y_pred=y_predict, target_names=iris.target_names))

prob = knc.predict_proba(x_test)
fpr, tpr, threhold = roc_curve(y_true=y_test, y_score=prob[:, 1], pos_label=1)
print(auc(fpr, tpr))

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, "red")
plt.show()

