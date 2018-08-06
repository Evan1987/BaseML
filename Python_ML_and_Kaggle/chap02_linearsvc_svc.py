
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, roc_curve, auc

digits = load_digits()
data = digits.data
label = digits.target

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=33)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
lsvc_y_predict = lsvc.predict(x_test)

svc = SVC(probability=True)
svc.fit(x_train, y_train)
svc_y_predict = svc.predict(x_test)
prob = svc.predict_proba(x_test)


# confusion matrix analysis for label “1”
fpr, tpr, threhold = roc_curve(y_true=y_test, y_score=prob[:, 1], pos_label=1)
df = pd.DataFrame({"actual": y_test, "pred": svc_y_predict})
newDF = df.applymap(lambda x: 1 if x == 1 else -1)
confusionMatrix = newDF.pivot_table(index="actual",
                                    columns="pred",
                                    fill_value=0,
                                    aggfunc=pd.Series.count)\
.sort_index(axis=0, ascending=False).sort_index(axis=1, ascending=False)

print(auc(fpr, tpr))

print("the ACC of Linear SVC is %f"%lsvc.score(x_test, y_test))

print(classification_report(y_true=y_test, 
                            y_pred=lsvc_y_predict, 
                            target_names=digits.target_names.astype(str)))

print(classification_report(y_true=y_test,
                            y_pred=svc_y_predict,
                            target_names=digits.target_names.astype(str)))

