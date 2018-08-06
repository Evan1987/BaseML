
# coding: utf-8
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

path = "F:/for learn/Python/python-machine-learning-book-master/code/datasets/movie/"
df = pd.read_csv(path + "movie_data.csv")

# 清洗函数
# 清洗掉 html语言， 留下表情符号
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' ' + ''.join(emotions).replace('-', '')
    return text

df["review"] = df["review"].apply(preprocessor)
X = df["review"].values
y = df["sentiment"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 普通分词（将文本按空格进行分词）
def tokenizer(text):
    return text.split()

# 在普通分词基础上提取相应词干
def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

# 去除停用词
stop = stopwords.words('english')
# print([w for w in tokenizer_porter("runners like running and thus they run a lot") if w not in stop])

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
lr_tfidf = Pipeline([("vect", tfidf), ("clf", LogisticRegression(random_state=0))])

#lr_tfidf.get_params()
params_grid = [{'vect__ngram_range': [(1, 1)],
                'vect__stop_words': [None, stop],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]}
              ]

# params_grid = [{'vect__ngram_range': [(1, 1)],
#                 'vect__stop_words': [None, stop],
#                 'vect__tokenizer': [tokenizer, tokenizer_porter],
#                 'clf__penalty': ['l1', 'l2'],
#                 'clf__C': [1.0, 10.0, 100.0]},
#                {'vect__ngram_range': [(1, 1)],
#                 'vect__stop_words': [None, stop],
#                 'vect__tokenizer': [tokenizer, tokenizer_porter],
#                 'vect__use_idf': [False],
#                 'vect__smooth_idf': [False],
#                 'vect__norm': [None],
#                 'clf__penalty': ['l1', 'l2'],
#                 'clf__C': [1.0, 10.0, 100.0]}
#               ]

def run():
    gs_lr_tfidf = GridSearchCV(estimator=lr_tfidf,
                               scoring="accuracy",
                               cv=3,
                               param_grid=params_grid,
                               n_jobs=1,
                               verbose=1)

    gs_lr_tfidf.fit(X_train, y_train)
    return gs_lr_tfidf


if __name__ == "__main__":
    gs_lr_tfidf = run()
    print("CV best ACC: %.3f" % gs_lr_tfidf.best_score_)
    print("Best estimator's param set: %s" % gs_lr_tfidf.best_params_)
    clf = gs_lr_tfidf.best_estimator_
    print("The test ACC : %.3f" % clf.score(X_test, y_test))