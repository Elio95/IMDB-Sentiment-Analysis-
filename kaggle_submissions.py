#!/usr/bin/env python3
from processing import data_no_features
from processing import write_submission

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV


raw = data_no_features()

def kaggle_21feb_ensemble_sub():
    tfv = TFIV(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=False,
            max_df=0.9)

    X_train, X_test, y_train, y_test = train_test_split(raw['x_raw'], raw['y'],
                                                        test_size=0.2, random_state=42)

    lr = LogisticRegression(solver='lbfgs') 
    lsvc = LinearSVC()
    nb = MultinomialNB()

    cp1 = Pipeline([('tfidf', tfv),('lsvc', lsvc)])
    cp2 = Pipeline([('tfidf', tfv),('lr', lr)])
    cp3 = Pipeline([('tfidf', tfv),('nb', nb)])

    # Hard voting: majority rules, Soft voting: weighted average probabilities
    eclf = VotingClassifier(estimators=[('lsvc', cp1), ('lr', cp2), ('nb', cp3)],
                        voting='hard')

    label = 'ensemble'

    scores = cross_val_score(eclf, X_train, y_train, cv=5, scoring='f1_micro')
    eclf.fit(X_train,y_train)
    mean_f1 = f1_score(y_test, eclf.predict(X_test), average='micro')
    print("Accuracy: %0.4f (+/- %0.4f) [%s] | Test: %0.4f" 
        % (scores.mean(), scores.std(), label, mean_f1))

    write_submission('21feb-ensemble-sub.csv',
                    raw['X_Ids'], eclf.predict(raw['X_raw']))

def kaggle_21feb_svm_max_df_filtering():
    tfv = TFIV(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=False,
            max_df=0.9)

    X_train, X_test, y_train, y_test = train_test_split(raw['x_raw'], raw['y'],
                                                        test_size=0.2, random_state=42)

    lsvc = LinearSVC(max_iter=100000)

    cp1 = Pipeline([('tfidf', tfv),('lsvc', lsvc)])


    label = 'svm'

    scores = cross_val_score(cp1, X_train, y_train, cv=5, scoring='f1_micro')
    cp1.fit(X_train,y_train)
    mean_f1 = f1_score(y_test, cp1.predict(X_test), average='micro')
    print("Accuracy: %0.4f (+/- %0.4f) [%s] | Test: %0.4f" 
        % (scores.mean(), scores.std(), label, mean_f1))

    write_submission('21feb-svm-max_df_filtering-sub.csv',
                    raw['X_Ids'], cp1.predict(raw['X_raw']))

kaggle_21feb_ensemble_sub()
kaggle_21feb_svm_max_df_filtering()
