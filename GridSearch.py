# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:01:34 2021

@author: Juli Gamboa
"""
# from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')

"""Para categorías de tiempo"""
# frutis=pd.read_csv("DATA/DATA_FINAL/1.DataFR_CT.csv")
# frutis=frutis.drop(["Time"], 1)
# X = np.array(frutis.drop(["Tiempo"], 1))
# y = np.array(frutis["Tiempo"])

"""Para tipo de muestra"""
frutis=pd.read_csv("Data/1.DataFR_OD_T.csv")
X = np.array(frutis.drop(["Tipo"], 1))
y = np.array(frutis["Tipo"])

# Split the dataset in two parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 20, 30, 40, 50, 60, 70, 80, 100]},
                    {'kernel': ['linear'], 'C': [1, 10, 20, 30, 40, 50, 60, 70, 80, 100]}]

scores = ['precision', 'recall', "f1"]

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

print(sklearn.metrics.SCORERS.keys())

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.