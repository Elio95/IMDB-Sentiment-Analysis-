#!/usr/bin/env python3

import PipelineClasses as pc
import pipelines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from processing import data_no_features


raw = data_no_features()

X_train, X_test, y_train, y_test = train_test_split(raw['x_raw'], raw['y'],
                                                    test_size=0.2, random_state=42)

tfv_opts = {'min_df':3, 'strip_accents':'unicode', 'token_pattern':r'\w{1,}',
           'ngram_range':(1, 1), 'sublinear_tf':False, 'max_df': 0.7}

# Pipes, kinda embarassed at the repetition here

uni = Pipeline([
    ('tfidf', TfidfVectorizer(**tfv_opts)),
    ('lsvc', LinearSVC(max_iter=100000)),
])
tfv_opts['ngram_range'] = (1,2)
bi = Pipeline([
    ('tfidf', TfidfVectorizer(**tfv_opts)),
    ('lsvc', LinearSVC(max_iter=100000)),
])
tfv_opts['ngram_range'] = (1,3)
tri = Pipeline([
    ('tfidf', TfidfVectorizer(**tfv_opts)),
    ('lsvc', LinearSVC(max_iter=100000)),
])
tfv_opts['ngram_range'] = (2,2)
bi_only = Pipeline([
    ('tfidf', TfidfVectorizer(**tfv_opts)),
    ('lsvc', LinearSVC(max_iter=100000)),
])
tfv_opts['ngram_range'] = (3,3)
tri_only = Pipeline([
    ('tfidf', TfidfVectorizer(**tfv_opts)),
    ('lsvc', LinearSVC(max_iter=100000)),
])
tfv_opts['ngram_range'] = (2,3)
bi_tri = Pipeline([
    ('tfidf', TfidfVectorizer(**tfv_opts)),
    ('lsvc', LinearSVC(max_iter=100000)),
])
# sublinear tfidf
tfv_opts['ngram_range'] = (1,2)
tfv_opts['sublinear_tf'] = True
bi_sublin = Pipeline([
    ('tfidf', TfidfVectorizer(**tfv_opts)),
    ('lsvc', LinearSVC(max_iter=100000)),
])
# sentiment
sent_pipe_ln = Pipeline([
    ('sentiment', pc.SentimentAnalyzerLinear()),
    ('sentiment_vect', pc.DictVectorizer()),
    ('lsvc', LinearSVC(max_iter=100000)),
])

sent_pipe_sq = Pipeline([
    ('sentiment', pc.SentimentAnalyzerSquared()),
    ('sentiment_vect', pc.DictVectorizer()),
    ('lsvc', LinearSVC(max_iter=100000)),
])

sent_pipe = Pipeline([
    ('sentiment', pc.SentimentAnalyzer()),
    ('sentiment_vect', pc.DictVectorizer()),
    ('lsvc', LinearSVC(max_iter=100000)),
])

pipes = {'uni': uni, 'bi':bi, 'tri':tri, 'bi_only':bi_only, 'tri_only':tri_only,
         'bi_tri': bi_tri, 'bi_sublin': bi_sublin, 'sentiment_lin':sent_pipe_ln,
         'sentiment_squared':sent_pipe_sq, 'sentiment_both':sent_pipe}

for k, v in pipes.items():
    eclf = v
    label = k
    scores = cross_val_score(eclf, X_train, y_train, cv=5, scoring='f1_micro')
    eclf.fit(X_train,y_train)

    mean_f1 = f1_score(y_test, eclf.predict(X_test), average='micro')

    print("Accuracy: %0.4f (+/- %0.4f) [%s] | Test: %0.4f" 
        % (scores.mean(), scores.std(), label, mean_f1))

