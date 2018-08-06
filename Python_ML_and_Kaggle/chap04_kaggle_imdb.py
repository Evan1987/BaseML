
# coding: utf-8
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


path = "F:/for learn/Python_ML_and_Kaggle/Datasets/imdb/"

train = pd.read_csv(path + "labeledTrainData.tsv", delimiter='\t')
test = pd.read_csv(path + "testData.tsv", delimiter='\t')

def review_to_text(review, remove_stopwords):
    ## 去掉 html标记
    raw_text = BeautifulSoup(review, 'html').get_text()
    ## 去掉非字母字符
    words = re.sub('[^a-zA-Z]', ' ', raw_text).lower().split()
    ## 去掉停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    return words


def fileGenerator(file, feature):
    output = []
    for review in file[feature]:
        words = review_to_text(review, True)
        text = ' '.join(words)
        output.append(text)
    return output

def subMission(result, file):
    sub = pd.DataFrame({'id': test['id'], 'sentiment': result})
    sub.to_csv(file, index=False)

x_train = fileGenerator(train, "review")
x_test = fileGenerator(test, "review")
y_train = train["sentiment"]


# 1. 分别采用 CountVectorizer, TfidfVectorizer进行贝叶斯训练
pipe_count = Pipeline([('count_vec', CountVectorizer(analyzer="word")),
                       ('mnb', MultinomialNB())])
pipe_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer="word")),
                       ('mnb', MultinomialNB())])

params_count = {'count_vec__binary': [True, False], 
                'count_vec__ngram_range': [(1, 1), (1, 2)], 
                'mnb__alpha': [0.1, 1.0, 10.0]}
params_tfidf = {'tfidf_vec__binary': [True, False],
                'tfidf_vec__ngram_range': [(1, 1), (1, 2)], 
                'mnb__alpha': [0.1, 1.0, 10.0]}

gs_count = GridSearchCV(estimator=pipe_count, 
                        param_grid=params_count, 
                        cv=4, 
                        n_jobs=-1, 
                        verbose=1)
gs_tfidf = GridSearchCV(estimator=pipe_tfidf, 
                        param_grid=params_tfidf, 
                        cv=4, 
                        n_jobs=-1, 
                        verbose=1)

gs_count.fit(x_train, y_train)
print(gs_count.best_params_)
print(gs_count.best_score_)
count_y_predict = gs_count.predict(x_test)


gs_tfidf.fit(x_train, y_train)
print(gs_tfidf.best_params_)
print(gs_tfidf.best_score_)
tfidf_y_predict = gs_tfidf.predict(x_test)


subMission(count_y_predict, path + "sub_count.csv")
subMission(tfidf_y_predict, path + "sub_tfidf.csv")


# 2. 用 word2vec进行预测
import nltk.data
from gensim.models import word2vec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


unlabeled_train = pd.read_csv(path + "unlabeledTrainData.tsv", delimiter='\t', quoting=3)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_to_text(raw_sentence, False))
    return sentences


corpora = []
for review in unlabeled_train['review']:
    corpora += review_to_sentences(review.encode('utf-8').decode('utf-8'), tokenizer)


num_features = 300
min_word_count = 20
num_workers = 4
context = 10
downsampling = 1e-3

model = word2vec.Word2Vec(sentences=corpora,
                          workers=num_workers,
                          size=num_features,
                          min_count=min_word_count, 
                          window=context, 
                          sample=downsampling)

model.init_sims(replace=True)
# model.save(path + "300features_20minwords_10context")
# model = Word2Vec.load(path + "300features_20minwords_10context")
# model.most_similar("man")


# 词向量产生文本特征向量
def makeFeatureVec(words, mdoel, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec += model[word]
    featureVec = featureVec/nwords
    return featureVec    


# 每个词条影评转化为基于词向量的特征向量（平均词向量）
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs


clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_text(review, remove_stopwords=True))

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_text(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

gbc = GradientBoostingClassifier()
params_gbc = {'n_estimators': [10, 100, 500],
              'learning_rate': [0.01, 0.1, 1.0],
              'max_depth': [2, 3, 4]}

gs = GridSearchCV(estimator=gbc, param_grid=params_gbc, cv=4, n_jobs=-1, verbose=1)
gs.fit(trainDataVecs, y_train)
print(gs.best_params_)
print(gs.best_score_)

wv_y_predict = gs.predict(testDataVecs)
subMission(wv_y_predict, path + "sub_wv.csv")

