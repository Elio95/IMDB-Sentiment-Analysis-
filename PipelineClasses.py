#!/usr/bin/env python3
# Format: Class followed by its pipeline

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from processing import data_no_features
# kinda extra
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

## Text Stats ad hoc

class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return [{'length': len(text),
                 'num_sentences': text.count('.')}
                for text in texts]

## VADER sentiment scores

class SentimentAnalyzer(BaseEstimator, TransformerMixin):
    """ Quick n dirty linear and nonlinear compound sentiment score
    """
    def __init__(self, squared=True, linear=True):
        self.squared = squared
        self.linear = linear

    def fit(self, x, y=None, squared=True):
        return self

    def transform(self, texts ):
        features = []

        analyser = SentimentIntensityAnalyzer()
        for text in texts:
            d = {}
            score = analyser.polarity_scores(text)
            compound = score['compound']
            if self.linear:
                d['sent'] = compound
            if self.squared:
                d['sent_squared'] = compound**2
            features.append(d)
        
        return features

class SentimentAnalyzerSquared(BaseEstimator, TransformerMixin):
    """ Quick n dirty linear and nonlinear compound sentiment score
    """
    def __init__(self, squared=True, linear=True):
        self.squared = squared
        self.linear = linear

    def fit(self, x, y=None, squared=True):
        return self

    def transform(self, texts ):
        features = []

        analyser = SentimentIntensityAnalyzer()
        for text in texts:
            d = {}
            score = analyser.polarity_scores(text)
            compound = score['compound']
            d['sent_squared'] = compound**2
            features.append(d)
        
        return features

class SentimentAnalyzerLinear(BaseEstimator, TransformerMixin):
    """ Quick n dirty linear and nonlinear compound sentiment score
    """
    def __init__(self, squared=True, linear=True):
        self.squared = squared
        self.linear = linear

    def fit(self, x, y=None, squared=True):
        return self

    def transform(self, texts ):
        features = []

        analyser = SentimentIntensityAnalyzer()
        for text in texts:
            d = {}
            score = analyser.polarity_scores(text)
            compound = score['compound']
            d['sent'] = compound
            features.append(d)
        
        return features
## DTM

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')

## Combining Features 

def assemble_feature_pipeline(clf, feats, clf_name='clf', **clf_args):
    """ return a feature pipeline consisting the custom features and the
    classifier passed as arguments

    clf: classifier 
    feats: set of features combined by FeatureUnion
    """
    return Pipeline([
        # Use FeatureUnion to combine the features
        ('union', feats),
        # Specify the classifier and args to use
        (str(clf_name), clf(clf_args)),
    ])

## Pipeline runtime

def main(seed=551):
    """ *breaths deeply* """
    raw = data_no_features()

    # pipeline = assemble_feature_pipeline(LinearSVC, custom_features, 'svm', max_iter=5000)

    X_train, X_test, y_train, y_test = train_test_split(raw['x_raw'], raw['y'],
                                                        test_size=0.2)

    scores = cross_val_score(custom_feature_pipeline, X_train, y_train, cv=3,
                             scoring='f1_micro')

    custom_feature_pipeline.fit(X_train,y_train)
    y_preds = custom_feature_pipeline.predict(X_test)

    mean_f1 = f1_score(y_test, y_preds, average='micro')

    print('CV scores:')
    for s in scores:
        print(s)

    print('\nF1 mean: ', mean_f1)

