
# coding: utf-8

# In[35]:

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
np.set_printoptions(precision=4)


# In[32]:

# 测试词袋模型（1元组模型， 默认）
docs = np.array(["The sun is shining", 
                 "The weather is sweet", 
                 "The sun is shining and the weather is sweet"])
count = CountVectorizer(ngram_range=(1, 1))
bags = count.fit_transform(docs)
print(count.vocabulary_)
print(bags.toarray())

# 将词袋模型的 vec结果（原始词频）对应上相应的词
vocab_index_dic = {v: k for k, v in count.vocabulary_.items()}
names = [vocab_index_dic.get(index) for index in range(len(vocab_index_dic))]
bags_df = pd.DataFrame(bags.toarray(), columns=names)
print(bags_df)


# In[53]:

# TF-IDF单词关联度
# nd:文档总数
# df(d, t)：包含词汇 t的文档 d的数量。
# tf(t, d)：词频统计
# tf-idf(t, d) = tf(t, d) * (idf(t, d) + 1)
# idf(t, d) = log((nd + 1) / (1 + df(d, t)))
# sklearn 还会对每个文档的结果做 L2归一化
tfidf = TfidfTransformer()
tfidf_df = pd.DataFrame(tfidf.fit_transform(bags).toarray(), columns=names)
print(tfidf_df)

