
# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer


sent1 = "The cat is walking in the bedroom."
sent2 = "A dog was running across the kitchen."
sents = [sent1, sent2]

count_vec = CountVectorizer()
print(count_vec.fit_transform(sents).toarray())
print(count_vec.get_feature_names())


import nltk
# ntlk的常用功能
# 对句子进行分割正规化
tokens1 = nltk.word_tokenize(sent1)
tokens2 = nltk.word_tokenize(sent2)
# 整理词汇表
vocab1 = sorted(set(tokens1))
vocab2 = sorted(set(tokens2))
# 寻找各词汇的原始词根
stemmer = nltk.stem.PorterStemmer()
stem1 = [stemmer.stem(t) for t in tokens1]
stem2 = [stemmer.stem(t) for t in tokens2]
print(stem1)
print(stem2)
# 初始化词性标注器，对每个词汇进行标注
pos_tag1 = nltk.tag.pos_tag(tokens1)
pos_tag2 = nltk.tag.pos_tag(tokens2)
print(pos_tag1)
print(pos_tag2)


from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk
import re


news = fetch_20newsgroups(subset="all")
x, y = news.data, news.target


# 将每条新闻中的句子逐一剥离， 并返回一个句子列表
def news_to_sent(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sent = tokenizer.tokenize(news_text)
    result = [re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split() for sent in raw_sent]
    return result


sentences = []
for i in x:
    sentences += news_to_sent(i)


# word2vec
from gensim.models import word2vec

# 词向量模型训练
model = word2vec.Word2Vec(sentences=sentences, 
                          workers=2, 
                          size=300, 
                          min_count=20, 
                          window=5, 
                          sample=1e-3)
# 表示当前训练好的词向量为最终版，可加快训练速度
model.init_sims(replace=True)
# 利用训练好的模型，寻找文本中与‘morning’,'email'最相关的 20个词汇
model.most_similar('morning', topn=20)
model.most_similar('email', topn=20)

