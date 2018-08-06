
# coding: utf-8
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)

# 生成距离矩阵
# pdist计算两两之间的距离， squareform生成对称阵
distMat = squareform(pdist(df, metric='euclidean'))
row_dist = pd.DataFrame(distMat, columns=labels, index=labels)

from scipy.cluster.hierarchy import linkage
# 输入到 linkage的可以是： ①距离矩阵上三角（pdist生成的结果）；②原始数据矩阵
# note： 不可以将 squareform生成的距离矩阵直接输入进去
# 生成关联矩阵（linkage matrix）
# wrong method ！!
print(linkage(row_dist, method='complete', metric='euclidean'))
print('\n')
# correct method !!
print(linkage(pdist(df, metric='euclidean'), method='complete'))
print('\n')
# correct method !!
print(linkage(df.values, method='complete', metric='euclidean'))

# 生成 linkage矩阵， 转化成数据框， 并绘树状图
row_clusters = linkage(df.values, method='complete', metric='euclidean')
hierarchy_df = pd.DataFrame(row_clusters, 
                            columns=['row_label 1', 'row_label 2', 'dist', 'num of items'], 
                            index=['cluster %d' % (i+1) for i in range(row_clusters.shape[0])])

hierarchy_df.head()

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean Dist')
plt.show()


# 绘制热度图
fig = plt.figure(figsize=(8, 8))
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='right')
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()


# 使用 sklearn进行层次聚类
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels : %s' % labels)

