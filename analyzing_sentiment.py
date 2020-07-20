# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:28:12 2020

@author: Hp
"""

import pandas as pd
dataset = pd.read_csv('amazon_baby.csv')
dataset = dataset[dataset['rating'] != 3]
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
dataset = dataset.reset_index()
import re
corpus = []
for i in range (0, 165679):
    review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if word in set(selected_words)]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() 
X = cv.fit_transform(corpus).toarray()
cv.vocabulary_

for j in range (0, 11):
    Sum = [sum(j) for j in zip(*X)]

dataset['sentiment'] = dataset['rating'] >= 4
Y = dataset.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state =0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
import numpy as np
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((165679, 1)).astype(int), values = X, axis = 1)
classifier_OLS = sm.OLS(endog = Y, exog = X).fit()
classifier_OLS.summary()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
(382 + 27764)/33136

Baby_Trend_Diaper_Champ = dataset.groupby(by = ['name']).get_group('Baby Trend Diaper Champ')
corpus_1 = []
for i in range (0, 298):
    review_1 = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
    review_1 = review_1.lower()
    corpus_1.append(review_1)
    
from sklearn.feature_extraction.text import CountVectorizer
cv_1 = CountVectorizer() 
X_1 = cv_1.fit_transform(corpus_1).toarray()
cv_1.vocabulary_(selected_words)

for j in range(0, 1):
    Sum = [sum(j) for j in zip(*Y_test)]

Y_1 = Baby_Trend_Diaper_Champ.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder_1 = LabelEncoder()
Y_1 = labelencoder_1.fit_transform(Y_1)

from sklearn.model_selection import train_test_split
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_1, Y_1, test_size = 0.2, random_state =0)
from sklearn.linear_model import LogisticRegression
classifier_1 = LogisticRegression()
classifier_1.fit(X_train_1, Y_train_1)

Y_pred_1 = classifier_1.predict_proba(X_test_1)
from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(Y_test_1, Y_pred_1)
46/60

classifier.predict()

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10, 50, 100], 'criterion': ['gini'], 'min_impurity_split': [1e-5, 1e-6, 1e-7, 1e-8], 'min_samples_leaf': [1, 10, 100, 1000]}, {'n_estimators': [10, 50, 100], 'criterion': ['entropy'], 'min_impurity_split': [1e-5, 1e-6, 1e-7, 1e-8], 'min_samples_leaf': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(estimator = classifier, cv = 10, n_jobs = -1, scoring = 'accuracy', param_grid = parameters)
grid_search = grid_search.fit(X_train, Y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_