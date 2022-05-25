# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:16:36 2021

@author: Juli Gamboa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

X =pd.read_csv("DataFR_CT_n4.C1.csv")
# d= {"Alto":"A", "Medio bajo": "MB", "Medio alto":"MA", "Corto" : "C"}
# X["Tiempo"] = X["Tiempo"].apply(lambda x:d[x])
X.head()
X=X.dropna()
true_labels=np.array(X["Tiempo"])
true_labels[:5]

features=X.drop(["Tiempo"], 1)
# data=np.array(X.drop(["Tipo"], 1))
features=np.array(features.drop(["SatN"], 1))
y=np.array(X["Tiempo"])
y
features[:5]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]

preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)

clusterer = Pipeline(
   [
       (
           "kmeans",
           KMeans(
               n_clusters=4,
               init="k-means++",
               n_init=50,
               max_iter=500,
               random_state=42,
           ),
       ),
   ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)

pipe.fit(scaled_features)

preprocessed_data = pipe["preprocessor"].transform(scaled_features)
predicted_labels = pipe["clusterer"]["kmeans"].labels_

silhouette_score(preprocessed_data, predicted_labels)

adjusted_rand_score(true_labels, predicted_labels)

adjusted_mutual_info_score(true_labels, predicted_labels)