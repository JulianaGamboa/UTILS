# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:00:57 2020

@author: Juli Gamboa
"""

import numpy as np
import pandas as pd
#df=pd.read_csv("Dataset_completo.csv")
df=pd.read_csv("Dataset_completo.csv")
df.columns

df=df.dropna()

# "#Dataset FR"
# df=df[df["Tipo"]=="FR"]

"#Dataset OD"
df=df[df["Tipo"]=="OD"]

############################################################
"ANÁLISIS EXPLORATORIO: quito NaN, quito Outliers (iqr*1.5)"
############################################################

"""PREPROCESSING"""
df["Tipo"].value_counts(normalize=True)
df["RetAreaHue"].describe()
IQR=df["RetAreaHue"].quantile(0.75)- df["RetAreaHue"].quantile(0.25)
IQR

"""QUITO OUTLIERS"""
#IQR*1.5 #define outliers
#X["AreaN"].quantile(0.75)+IQR*1.5 #define outliers
#X["AreaN"].quantile(0.25)-IQR*1.5 #define outliers
df=df[df["RetAreaHue"]<df["RetAreaHue"].quantile(0.75)+IQR*1.5]
df=df[df["RetAreaHue"]>df["RetAreaHue"].quantile(0.25)-IQR*1.5]

IQR=df["RetBrig"].quantile(0.75)- df["RetBrig"].quantile(0.25)
IQR
df=df[df["RetBrig"]<df["RetBrig"].quantile(0.75)+IQR*1.5]
df=df[df["RetBrig"]>df["RetBrig"].quantile(0.25)-IQR*1.5]

IQR=df["RetSat"].quantile(0.75)- df["RetSat"].quantile(0.25)
df=df[df["RetSat"]<df["RetSat"].quantile(0.75)+IQR*1.5]
df=df[df["RetSat"]>df["RetSat"].quantile(0.25)-IQR*1.5]

IQR=df["RetAreaRec"].quantile(0.75)- df["RetAreaRec"].quantile(0.25)
IQR
df=df[df["RetAreaRec"]<df["RetAreaRec"].quantile(0.75)+IQR*1.5]
df=df[df["RetAreaRec"]>df["RetAreaRec"].quantile(0.25)-IQR*1.5]

IQR=df["RetWE"].quantile(0.75)- df["RetWE"].quantile(0.25)
df=df[df["RetWE"]<df["RetWE"].quantile(0.75)+IQR*1.5]
df=df[df["RetWE"]>df["RetWE"].quantile(0.25)-IQR*1.5]
IQR

IQR=df["RetHE"].quantile(0.75)- df["RetHE"].quantile(0.25)
df=df[df["RetHE"]<df["RetHE"].quantile(0.75)+IQR*1.5]
df=df[df["RetHE"]>df["RetHE"].quantile(0.25)-IQR*1.5]

IQR=df["RetRA"].quantile(0.75)- df["RetRA"].quantile(0.25)
df=df[df["RetRA"]<df["RetRA"].quantile(0.75)+IQR*1.5]
df=df[df["RetRA"]>df["RetRA"].quantile(0.25)-IQR*1.5]

df.to_excel("df_frutillas_sin_outliers.xlsx")

"""###################################
###PRINCIPAL COMPONENTS ANALYSIS###
###################################"""

from sklearn.preprocessing import StandardScaler
features = ['RetSat', 'RetBrig',
       'RetAreaHue', 'RetAreaRec', 'RetRA', 'RetWE', 'RetHE'
       ]
sample = ['Tipo_', 'Tipo',
       'Time', 'Tiempo',
       "Coating"]
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, sample].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', "principal component 3"])

print(principalComponents)

# ##Suma de varianza explicada por el n_components elegido
np.cumsum(pca.explained_variance_ratio_)

finalDf = pd.concat([principalDf, df[sample]], axis = 1)

##############
###HEAT_MAP###
##############

import seaborn as sns
ax = sns.heatmap(pca.components_,
                 cmap='YlGnBu',
                 yticklabels=[ "PCA"+str(x) for x in range(1,pca.n_components_+1)],
                 xticklabels=list(features),
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")

##############
####BIPLOT####
##############

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 14})
labels = features
txt = df["Tipo"]
def myplot(score,coeff,labels= labels):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs* scalex,ys * scaley, color = "white", s=5)
        # plt.scatter(xs*0.1,ys*0.1, color = "black", s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'gray',alpha = 0.5)
        # if labels is None:
        #     plt.text(coeff[i,0]* 0.85, coeff[i,1] * 0.85, "Var"+str(i+1), color = 'gray', ha = 'center', va = 'center')
        # else:
        plt.text(coeff[i,0]* 0.85, coeff[i,1] * 0.85, labels[i], color = 'black', ha = 'center', va = 'center') 
    plt.xlabel("PC1 (61.2%)", fontsize = 15)
    plt.ylabel("PC2 (79.9%)", fontsize = 15)
    plt.grid()

myplot(principalComponents[:,0:2],np.transpose(pca.components_[0:2, :]),labels)
plt.show()

print(principalComponents)
print(np.transpose(pca.components_[0:2, :])) #coordenadas vectores

#Scatter plot (Score plot)
PC1 = finalDf["principal component 1"]
x = finalDf.loc[:, "principal component 1"].values
PC2 = finalDf["principal component 2"]
y = finalDf.loc[:, "principal component 2"].values
PC3 = finalDf["principal component 3"]
z = finalDf.loc[:, "principal component 3"].values

# label = finalDf["Samples"]
# txt = finalDf.loc[:, "Samples"].values
fig, ax = plt.subplots(1, figsize=(10, 6))
fig.suptitle('Score plot')
ax.scatter(x, y, color="black")
ax.set_xlabel('PC1 (61.2%)', fontsize = 15)
ax.set_ylabel('PC2 (79.9%)', fontsize = 15)
plt.grid()
for i, txt in enumerate(txt):
    ax.annotate(txt, (x[i], y[i]))
# label = finalDf["Samples"]
# txt = finalDf.loc[:, "Samples"].values

"#########################"
"""KMEANS CLUSTERING"""
"#########################"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from kneed import KneeLocator
# from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

###POR TIPO###

true_labels_tipo=y[:, 1]
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_labels_encoded_tipo=le.fit_transform(true_labels_tipo)
print(True_labels_encoded_tipo)
True_labels_encoded_tipo.shape

###POR CATEGORÍA DE TIEMPO###
true_labels_cat_tiempo=y[:, 3]
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_labels_encoded_CT=le.fit_transform(true_labels_cat_tiempo)
print(True_labels_encoded_CT)
True_labels_encoded_CT.shape

###POR PRESENCIA DE COATING###
true_labels_coating=y[:, 4]
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_labels_encoded_coat=le.fit_transform(true_labels_coating)
print(True_labels_encoded_coat)
True_labels_encoded_coat.shape

###INICIALIZO K-MEANS###
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
kmeans = KMeans(n_clusters=5, **kmeans_kwargs)
kmeans.fit(x)
kmeans.labels_.shape
kmeans_labels=kmeans.labels_

ari = round(adjusted_rand_score(True_labels_encoded_tipo, kmeans.labels_), 2)
ami = round(adjusted_mutual_info_score(True_labels_encoded_tipo, kmeans.labels_), 2)
print("For dataset the results of kmeans clustering are: ")
print("ari_: ", ari)
print("ami_: ", ami)

###CÁLCULO DE SILHOUETTE_COEFFICIENTS###
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x)
    score = silhouette_score(x, kmeans.labels_)
    silhouette_coefficients.append(score)
    
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

from sklearn.cluster import KMeans
n = 6
k_means = KMeans(n_clusters=n)
k_means.fit(x)

centroides = k_means.cluster_centers_
etiquetas = k_means.labels_
etiquetas
centroides

plt.plot(x[etiquetas==0,0], x[etiquetas==0,1], "m.", label='cluster 1')
plt.plot(x[etiquetas==1,0], x[etiquetas==1,1], "r.", label='cluster 2')
plt.plot(x[etiquetas==2,0], x[etiquetas==2,1], "y.", label='cluster 3')
plt.plot(x[etiquetas==1,0], x[etiquetas==1,1], "b.", label='cluster 4')
plt.plot(x[etiquetas==2,0], x[etiquetas==2,1], "g.", label='cluster 5')
plt.plot(x[etiquetas==2,0], x[etiquetas==2,1], "k.", label='cluster 6')

plt.plot(centroides[:,0],centroides[:,1],'c.',markersize=15, label='centroides')

plt.legend(loc='best')
plt.show()

""""""""""""""""""""""""""
"""###Silhuette Plots###"""
""""""""""""""""""""""""""

from sklearn.metrics import silhouette_samples
for i, k in enumerate([2, 3, 4, 5]): #nro de clusters a considerar
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # Run the Kmeans algorithm
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(x)
    centroids = km.cluster_centers_

    # Get silhouette samples
    silhouette_vals = silhouette_samples(x, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    #ax1.set_title('Silhouette plot for the various clusters', y=1.02);
    
    # Scatter plot of data colored with labels
    ax2.scatter(x[:, 0], x[:, 1], c=labels)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='+', c='r', s=1000)
    ax2.set_xlim([-2, 2])
    ax2.set_xlim([-2, 2])
    #ax2.set_xlabel('PC1')
    #ax2.set_ylabel('PC2')
    #ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}',
                 fontsize=16, fontweight='semibold', y=1.05);

############
###DBSCAN###
############

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

true_labels=df["Tiempo"]

#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_labels_encoded=le.fit_transform(true_labels)
print(True_labels_encoded)
True_labels_encoded.shape

X = principalComponents
db = DBSCAN(eps=1.2, min_samples=2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: ', n_clusters_)
print('Estimated number of noise points: ', n_noise_)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: ' % n_clusters_)
plt.show()

ari = adjusted_rand_score (True_labels_encoded, labels)
ami = adjusted_mutual_info_score (True_labels_encoded, labels)
score = silhouette_score(principalComponents, labels)
print("silhouette_score: ", score)
print("ari: ", ari)
print("ami: ", ami)

"""########################
###CLUSTERS DE TIEMPO######
###########################"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
#df=pd.read_csv("Dataset_completo.csv")
# df=pd.read_csv("df_frutillas_sin_outliers_means.csv")
# df.columns
# df=df.dropna()
# df=df[df["Tipo_"]==0]

#df=finalDf[finalDf["Tipo_"]==1]
df=df[df["Tipo_"]==1]
from sklearn.preprocessing import StandardScaler
features = ['RetSat', 'RetBrig',
        'RetAreaHue', 'RetAreaRec', 'RetRA', 'RetWE', 'RetHE'
        ]
#features = ["principal component 1", "principal component 2", "principal component 3"]
sample = ['Tipo',
       'Time', 'Tiempo', "Coating"
       ]
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, sample].values
# Standardizing the features
# x = StandardScaler().fit_transform(x)
# print(x)

true_labels=df["Tiempo"]

#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_labels_encoded=le.fit_transform(true_labels)
print(True_labels_encoded)
True_labels_encoded.shape

""""""""""""
"""KMEANS"""
""""""""""""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from kneed import KneeLocator
# from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
kmeans.fit(x)
kmeans.labels_.shape

ari = round(adjusted_rand_score(True_labels_encoded, kmeans.labels_), 2)
ami = round(adjusted_mutual_info_score(True_labels_encoded, kmeans.labels_), 2)
print("For dataset the results of kmeans clustering are: ")
print("ari_: ", ari)
print("ami_: ", ami)

silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x)
    score = silhouette_score(x, kmeans.labels_)
    silhouette_coefficients.append(score)
    
plt.style.use("fivethirtyeight")
plt.plot(range(2, 10), silhouette_coefficients)
plt.xticks(range(2, 10))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

from sklearn.cluster import KMeans
n = 7
k_means = KMeans(n_clusters=n)
k_means.fit(x)

centroides = k_means.cluster_centers_
etiquetas = k_means.labels_
etiquetas
centroides

plt.plot(x[etiquetas==0,0], x[etiquetas==0,1],'b.', label='cluster 1')
plt.plot(x[etiquetas==1,0], x[etiquetas==1,1],'r.', label='cluster 2')
plt.plot(x[etiquetas==2,0], x[etiquetas==2,1],'y.', label='cluster 3')
plt.plot(x[etiquetas==3,0], x[etiquetas==3,1],'k.', label='cluster 4')
plt.plot(x[etiquetas==4,0], x[etiquetas==4,1],'g.', label='cluster 5')
plt.plot(x[etiquetas==3,0], x[etiquetas==3,1],'o', label='cluster 6')
plt.plot(x[etiquetas==4,0], x[etiquetas==4,1],'s', label='cluster 7')
plt.plot(centroides[:,0],centroides[:,1],'m.',marker= "+",  markersize=25, label='centroids')

plt.legend(loc='best')
plt.show()

""""""""""""""""""""""""""
"""###Silhuette Plots###"""
""""""""""""""""""""""""""
df=pd.read_csv("df_frutillas_sin_outliers.csv")
df=df[df["Tipo_"]==0]

from sklearn.preprocessing import StandardScaler
features = ['RetSat', 'RetBrig',
        'RetAreaHue', 'RetAreaRec', 'RetRA', 'RetWE', 'RetHE'
        ]
sample = ['Tipo',
        'Time', 'Tiempo'
        ]
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, sample].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)

true_labels=df["Tiempo"]

#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_labels_encoded=le.fit_transform(true_labels)
print(True_labels_encoded)
True_labels_encoded.shape

from sklearn.metrics import silhouette_samples
for i, k in enumerate([2, 3, 4]): #nro de clusters a considerar
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # Run the Kmeans algorithm
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(x)
    centroids = km.cluster_centers_

    # Get silhouette samples
    silhouette_vals = silhouette_samples(x, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);
    
    # Scatter plot of data colored with labels
    ax2.scatter(x[:, 0], x[:, 1], c=labels)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='+', c='r', s=1000)
    ax2.set_xlim([-2, 2])
    ax2.set_xlim([-2, 2])
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {k}',
                 fontsize=16, fontweight='semibold', y=1.05);

############
###DBSCAN###
############

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#df=pd.read_csv("Dataset_completo.csv")
df=pd.read_csv("df_frutillas_sin_outliers_means.csv")
df.columns
df=df[df["Tipo_"]==0]
true_labels=df["Tiempo"]

#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_labels_encoded=le.fit_transform(true_labels)
print(True_labels_encoded)
True_labels_encoded.shape

from sklearn.preprocessing import StandardScaler
features = ['RetSat', 'RetBrig',
       'RetAreaHue', 'RetAreaRec', 'RetRA', 'RetWE', 'RetHE'
       ]
sample = ['Tipo',
       'Time', 'Tiempo'
       ]
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, sample].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)

X = x
db = DBSCAN(eps=1.4, min_samples=2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: ', n_clusters_)
print('Estimated number of noise points: ', n_noise_)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
plt.show()

ari = adjusted_rand_score (True_labels_encoded, labels)
ami = adjusted_mutual_info_score (True_labels_encoded, labels)
score = silhouette_score(X, labels)
print("silhouette_score: ", score)
print("ari: ", ari)
print("ami: ", ami)