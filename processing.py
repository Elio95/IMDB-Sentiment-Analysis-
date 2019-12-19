#!/usr/bin/env python3
""" 1 week in: prime time to finally make a (more) readable and repetable
processing 'pipeline'. Read the docstring of main() for a howto

csr_matrix will be the fundamental datatype here
"""

import nltk
import os
from os.path import split as ps
from os.path import join as pj
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import csr_matrix
import pickle

def main(fun, **kwargs):
    """ Generate feature matricies from raw labeled (train/valid) and unlabeled
    (test/submission) data using the function passed as the positional arg. By
    swapping out the feature function used we can make a kind of bootleg custom
    pipelines (by using passing the results of this to models for training and
    CV)
    Returns a dictionary of the form:
        d = {'x': csr_matrix # sparse feature matrix for labeled data
             'y': list # the classes corresponding to each row in d['x']
             'X': csr_matrix # sparse feature matrix for unlabeled data
             'X_Ids': list # the file Ids corresponding to each row in d['X']
        }

    Data directories are hard coded. The assumed structure (relative to where
    this code is being run from) is:
        data/train/neg # labeled data (class 0)
        data/train/pos # labeled data (class 1)
        data/test      # unlabeled data

    args:
        fun: the feature_generating function to use. Should have these properties:
            - accept texts, corpus (lists of documents) as its positional args
            - return a scipy.sparse.csr_matrix
        **kwargs: pass any keyword (optional) args to fun.
    """

    # Load labeled training data
    neg = read_txt('data/train/neg')
    pos = read_txt('data/train/pos')
    train_texts = [d['text'] for d in neg + pos]
    y = [0 for i in range(len(neg))]
    y = y + [1 for i in range(len(pos))]

    # Load unlabeled data
    test_dir = 'data/test'
    comments = read_txt(test_dir)
    test_texts = [d['text'] for d in comments]
    X_Ids = [d['Id'] for d in comments]

    corpus = [text for text in train_texts]
    print('Features for x (labeled)')
    x = fun(train_texts, corpus, dataset='train', **kwargs)
    print('Features for X (unlabeled)')
    X = fun(test_texts, corpus, dataset='test', **kwargs)

    d =  {'x':x, 'y':y, 'X':X, 'X_Ids':X_Ids}
    return d

def main_50k_corpus(fun, **kwargs):
    """ Generate feature matricies from raw labeled (train/valid) and unlabeled
    (test/submission) data using the function passed as the positional arg. By
    swapping out the feature function used we can make a kind of bootleg custom
    pipelines (by using passing the results of this to models for training and
    CV)
    Returns a dictionary of the form:
        d = {'x': csr_matrix # sparse feature matrix for labeled data
             'y': list # the classes corresponding to each row in d['x']
             'X': csr_matrix # sparse feature matrix for unlabeled data
             'X_Ids': list # the file Ids corresponding to each row in d['X']
        }

    Data directories are hard coded. The assumed structure (relative to where
    this code is being run from) is:
        data/train/neg # labeled data (class 0)
        data/train/pos # labeled data (class 1)
        data/test      # unlabeled data

    args:
        fun: the feature_generating function to use. Should have these properties:
            - accept texts, corpus (lists of documents) as its positional args
            - return a scipy.sparse.csr_matrix
        **kwargs: pass any keyword (optional) args to fun.
    """

    # Load labeled training data
    neg = read_txt('data/train/neg')
    pos = read_txt('data/train/pos')
    train_texts = [d['text'] for d in neg + pos]
    y = [0 for i in range(len(neg))]
    y = y + [1 for i in range(len(pos))]

    # Load unlabeled data
    test_dir = 'data/test'
    comments = read_txt(test_dir)
    test_texts = [d['text'] for d in comments]
    X_Ids = [d['Id'] for d in comments]

    # corpus = [text for text in train_texts]
    corpus = expanded_corpus()
    print('Features for x (labeled)')
    x = fun(train_texts, corpus, dataset='train', **kwargs)
    print('Features for X (unlabeled)')
    X = fun(test_texts, corpus, dataset='test', **kwargs)

    d =  {'x':x, 'y':y, 'X':X, 'X_Ids':X_Ids}
    return d

def read_txt(ddir):
    """ Read all .txt files in dir into a list(dict) w/ Id (from filename) and
    text keys
    """
    comments = []
    files = os.listdir(ddir)
    # sort files in ascending numeric order
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    for filename in files:
        with open(pj(ddir,filename), 'rb') as f:
            d = {'text': f.read().decode('utf-8').replace('\n', '')}
            Id = filename.split('.')[0]
            d['Id'] = ps(Id)[-1]
            comments.append(d)
    return comments

def features(texts, corpus, mf=None, mdf=1, dataset=None, pickled_sentiment=False):
    """ bias, tfidf(words), tfidf(bigrams), sentiment, length squared
    """
    X = term_frequencies(texts, corpus, max_df=mdf, max_features=mf)

    feats = []
    feats.append(term_frequencies(texts, corpus, max_df=mdf, max_features=mf,
                                  ngram_range=(2,2)))
    feats.append(length_features(texts, squared_only=True))
    if pickled_sentiment:
        feats.append(load_pickle_wrapper(dataset, 'compound_sentiment.pickle'))
    else:
        feats.append(sentiment(texts, compound_only=True))

    # bias term
    feats.append(csr_matrix([1 for i in range(len(texts))]))

    for f in feats:
        X = sparse_hstack(X, f)
    return X

def features_elio_20feb(texts, corpus, dataset, pickled_sentiment=False):
    """ bias, tfidf(words), tfidf(bigrams), sentiment, length squared
    """
    X = term_frequencies(texts, corpus, min_df=3, strip_accents='unicode',
                         analyzer='word',token_pattern=r'\w{1,}', 
                         ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                         sublinear_tf=True, stop_words = 'english')

    feats = []
    feats.append(length_features(texts, squared_only=True))
    if pickled_sentiment:
        feats.append(load_pickle_wrapper(dataset, 'compound_sentiment.pickle'))
    else:
        feats.append(sentiment(texts))

    # bias term
    feats.append(csr_matrix([1 for i in range(len(texts))]))

    for f in feats:
        X = sparse_hstack(X, f)
    return X

def features_words(texts, corpus, dataset=None, mf=None, mdf=1.0, pickled_sentiment=False):
    """ bias, tfidf(words, no bigrams), compound sentiment, nchar²
    """
    X = term_frequencies(texts, corpus, max_df=mdf, max_features=mf)

    feats = []
    feats.append(length_features(texts, squared_only=True))

    if pickled_sentiment:
        feats.append(load_pickle_wrapper(dataset, 'compound_sentiment.pickle'))
    else:
        feats.append(sentiment(texts, compound_only=True))
    # bias term
    feats.append(csr_matrix([1 for i in range(len(texts))]))

    for f in feats:
        X = sparse_hstack(X, f)
    return X

def features_nosent(texts, corpus, mf=None, mdf=1):
    """ So we can treat x and X the same and design feature sets
    """
    X = term_frequencies(texts, corpus, max_df=mdf, max_features=mf)

    feats = []
    feats.append(length_features(texts))

    feats.append(term_frequencies(texts, corpus, max_df=mdf, max_features=mf,
                                  ngram_range=(2,2)))
    # bias term
    feats.append(csr_matrix([1 for i in range(len(texts))]))

    for f in feats:
        X = sparse_hstack(X, f)
    return X

def features_nosent_bigrams(texts, corpus, mf=None, mdf=0.9):
    """ So we can treat x and X the same and design feature sets
    """
    X = term_frequencies(texts, corpus, max_df=mdf, max_features=mf,
                                  ngram_range=(2,2))
    feats = []
    feats.append(length_features(texts))

    # bias term
    feats.append(csr_matrix([1 for i in range(len(texts))]))

    for f in feats:
        X = sparse_hstack(X, f)
    return X

def features_binary(texts, corpus, dataset=None, pickled_sentiment=False, **kwargs):
    """ Binary feature matrix for bern. bayes.
    Features:
    """
    # Start w/ DTM, since it will never be a 1d array
    X = term_frequencies(texts, corpus, binary=True, **kwargs)
    
    feats = []

    if pickled_sentiment:
        feats.append(load_pickle_wrapper(dataset, 'compound_sentiment.pickle'))
    else:
        feats.append(sentiment(texts, compound_only=True))
    for f in feats:
        X = sparse_hstack(X, f)
    return X.astype(int)

def load_pickle_wrapper(dataset, suffix):
    """ Wrapper to check if dataset is specified when loading pickle file
    in feature pipeline. The file loaded is generated from dataset and suffix.
    The suffix should include the .pickle extension
        <dataset>-<suffix>

    args:
        dataset: str, one of train or test
        suffix:  str, the rest of the filename, including extension. 
            eg 'compound_sentiment.pickle'
    """
    if dataset == None:
        sys.exit('Err: tried to load pickle sentiment but no dataset arg '\
                    + 'please specify dataset= "test" or "train"')
    # If loading pickle, must specify whether this is training or test data
    pickle_file = pj('data/pickle', '-'.join((dataset, suffix)))
    return load_pickle(pickle_file)

def sparse_hstack(X, X_add):
    """ Try transposing if first hstack fails
    """
    try:
        return scipy.sparse.hstack((X, X_add))
    except ValueError:
        print('Transposing 1d matrix before hstack')
        return scipy.sparse.hstack((X, X_add.transpose()))

def length_features(texts, squared_only=False):
    """ return feature vector of scaled (/mean) [n_chars and/or n_chars_squared]
    """
    # Calc mean length
    sum_len = 0
    for text in texts:
        sum_len = sum_len + len(text)
    mean_len = sum_len / len(texts)

    if squared_only:
        # still return single dim feature as a list since downstream expects it
        feature = [[(len(text)/mean_len)**2 for text in texts]]
    else:
        feature = [[len(text)/mean_len, (len(text)/mean_len)**2] for text in texts]
    return csr_matrix(feature)

def sentiment(texts, binary_only=False, include_neutral=True,
              compound_only=False): 
    """ Quick and dirty sentiment analyser.  

    Switches:
        binary_only : only return binary scores if true (is_positive, is_negative)

        include_neutral : whether or not to encode for neutral sentiment
        (-0.05<sentiment<0.05) texts. 
            True: is_positive, is_negative
            False: is_positive only (1 if sentiment>0, else 0)

    Outputs
        sentiment: the compound lexicon score
        is_positive, is_negative: binary encoded, if both are 0 text was rated
        neural

        `analyser.polarity_scores(sentence)` gives us scores on 4 sentiment metrics:
            1. neg
            2. neu
            3. pos
            4. compound
                - sum of all lexicon ratings
                - positive: compound > 0.05
                - neutral: 0.05 > compound > -0.05
                - negative: -0.05 > compound
        
            neg, neu, and pos are the % of the text which were rated as each
            valence and should at to 1
    """
    analyser = SentimentIntensityAnalyzer()

    out = []
    print('Compute sentiment features')
    for text in tqdm(texts):
        score = analyser.polarity_scores(text)
        compound = score['compound']
        is_positive = 0
        is_negative = 0
        if include_neutral:
            if compound > 0.05:
                is_positive = 1
            elif compound < -0.05:
                is_negative = 1
            if binary_only:
                feature = [is_positive, is_negative]
            else:
                feature =  [compound, is_positive, is_negative]
        else:
            # include_neutral is False
            if compound > 0:
                is_positive = 1
            if binary_only:
                feature =  [is_positive]
            else:
                feature = [compound, is_positive]
        if compound_only:
            feature = compound
        out.append(feature)
    return csr_matrix(out)

def term_frequencies(texts, corpus, **kwargs):
    """ Return a document term matrix (DTM) in a sparse matrix format

    text : a list of the documents to be transformed to DTM

    corpus : a list of documents from which to learn vocabulary and IDF

    **kwargs : kwargs for TfidfVectorizer
    """
    vec = TfidfVectorizer(kwargs)
    # Learn vocabulary and idf from training set
    vec.fit(corpus)

    # Transform documents to document-term matrix
    return vec.transform(texts)

### 20 Feb search for additional features

def expanded_corpus():
    """ Generate a corpus from the Large Movie Review Dataset, ignoring any
    labels. Since this is likely the dataset used for the competition, I'm not
    sure if this is cool or not, but it will let us test if a bigger corpus 
    improves our models and, if so, we can scrape data from IMDB ourselves.
    """
    dirs = [pj('data/corpus/', x) for x in ['neg', 'neg2', 'pos', 'pos2']]
    corpus = []
    for x in dirs:
        docs = read_txt(x)
        corpus = corpus + [i['text'] for i in docs]
    return corpus

### Save pickle object(s) for later

def pickle_objects():
    """ Pickle csr_matricies of long-to-compute features (looking at you,
    sentiment) to be loaded, and then stacked, by feature functions for faster
    execution
    """
    save_dir='data/pickle'

    # Load labeled training data
    neg = read_txt('data/train/neg')
    pos = read_txt('data/train/pos')
    train_texts = [d['text'] for d in neg + pos]

    # Load unlabeled data
    test_dir = 'data/test'
    comments = read_txt(test_dir)
    test_texts = [d['text'] for d in comments]

    # Compound sentiment
    name_texts = [('train', train_texts), ('test', test_texts)]
    for tup in name_texts:
        if not os.path.exists(save_dir):
            os.makedirs(dir)
        outfile = pj(save_dir, tup[0] + '-compound_sentiment.pickle')
        save_pickle(outfile, sentiment(tup[1], compound_only=True))

### Convinience

def write_submission(outfile, Ids, Y_pred):
    """ Write Id,Category csv for submission to kaggle
    """
    with open(outfile, 'w') as f:
        f.write('Id,Category\n')
        for i in range(len(Ids)):
            f.write(Ids[i])
            f.write(',')
            f.write(str(int(Y_pred[i])))
            f.write('\n')

def save_pickle(filename, obj):
    """ Save a python object to disk
    pickle files usually have a .p or .pickle extension
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(pickle_file):
    """ Load a pickled (saved) python object
    """
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def data_no_features():
    # Load labeled training data
    neg = read_txt('data/train/neg')
    pos = read_txt('data/train/pos')
    train_texts = [d['text'] for d in neg + pos]
    y = [0 for i in range(len(neg))]
    y = y + [1 for i in range(len(pos))]

    # Load unlabeled data
    test_dir = 'data/test'
    comments = read_txt(test_dir)
    test_texts = [d['text'] for d in comments]
    X_Ids = [d['Id'] for d in comments]

    return {'x_raw': train_texts, 'y': y, 'X_raw': test_texts, 'X_Ids': X_Ids}

