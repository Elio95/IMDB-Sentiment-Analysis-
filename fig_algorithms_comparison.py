#!/usr/bin/env python3

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from processing import data_no_features

raw = data_no_features()


tfv = TFIV(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), sublinear_tf=False,
        max_df=0.9)

X_train, X_test, y_train, y_test = train_test_split(raw['x_raw'], raw['y'],
                                                    test_size=0.2, random_state=42)

lr = LogisticRegression(solver='lbfgs') 
lsvc = LinearSVC()
nb = MultinomialNB()
dt = DecisionTreeClassifier()

cp1 = Pipeline([('tfidf', tfv),('lsvc', lsvc)])
cp2 = Pipeline([('tfidf', tfv),('lr', lr)])
cp3 = Pipeline([('tfidf', tfv),('nb', nb)])
cp4 = Pipeline([('tfidf', tfv),('dt', dt)])

# Hard voting: majority rules, Soft voting: weighted average probabilities
eclf = VotingClassifier(estimators=[('lsvc', cp1), ('lr', cp2), ('nb', cp3)],
                        voting='hard')
eclf2 = VotingClassifier(estimators=[('lsvc', cp1), ('lr', cp2), ('nb', cp3),
                                     ('dt', cp4)], voting='hard')

models = {}
for clf, label in zip([eclf], ['LSVC', 'LogReg', 'naive Bayes',
                               'Decision Trees', 'Ensemble no DT', 'Ensemble all']):
    print('Iter:', label)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_micro')
    clf.fit(X_train,y_train)
    mean_f1 = f1_score(y_test, clf.predict(X_test), average='micro')
    print("Accuracy: %0.4f (+/- %0.4f) [%s] | Test: %0.4f" 
          % (scores.mean(), scores.std(), label, mean_f1))
   
