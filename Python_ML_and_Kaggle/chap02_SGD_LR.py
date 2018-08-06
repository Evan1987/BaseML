
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc

path = "F:/for learn/Python_ML_and_Kaggle/Datasets/Breast-Cancer/"

colnames = ["Sample code number", 
            "Clump Thickness", 
            "Uniformity of Cell Size", 
            "Uniformity of Cell Shape", 
            "Marginal Adhesion", 
            "Single Epithelial", 
            "Bare Nuclei", 
            "Bland Chromatin", 
            "Normal Nucleoli", 
            "Mitoses", 
            "Class"]

data = pd.read_csv(path + "breast-cancer-wisconsin.data", names=colnames)
# data.apply(lambda x: np.sum(x == "?"), axis=0)

data = data.replace(to_replace="?", value=np.nan)
data = data.dropna(how="any")
# data.shape
x_train, x_test, y_train, y_test = train_test_split(data[colnames[1:10]],
                                                    data[colnames[10]],
                                                    test_size=0.25,
                                                    random_state=33)

y_train.value_counts()
y_test.value_counts()

# 特征标准化，根据训练集fit的信息才能transform测试集。
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

lr = LogisticRegression()
sgd = SGDClassifier()

lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
sgd.fit(x_train, y_train)
sgd_y_predict = sgd.predict(x_test)
# lr.score(x_test,y_test)
# sgd.score(x_test,y_test)

print(classification_report(y_pred=lr_y_predict, y_true=y_test, target_names=["Benign", "Maglignant"]))

fpr, tpr, threhold = roc_curve(y_true=y_test, y_score=lr.predict_proba(x_test)[:, 0], pos_label=2)
print(auc(fpr, tpr))

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, "red")
plt.show()