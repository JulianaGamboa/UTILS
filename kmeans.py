# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:51:27 2020

@author: Juli Gamboa
"""
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


#x1 = np.random.standard_normal((100,2))*0.6+np.ones((100,2))
#x2 = np.random.standard_normal((100,2))*0.5-np.ones((100,2))
#x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
#X = np.concatenate((x1,x2,x3),axis=0)

# X =pd.read_csv("DataFR_OD_tipo.csv")
# # d= {"Alto":"A", "Medio bajo": "MB", "Medio alto":"MA", "Corto" : "C"}
# # X["Tiempo"] = X["Tiempo"].apply(lambda x:d[x])
# X.head()
# data=np.array(X.drop(["Tipo"], 1))

X =pd.read_csv("DataFin_1_s.csv")
# d= {"Alto":"A", "Medio bajo": "MB", "Medio alto":"MA", "Corto" : "C"}
# X["Tiempo"] = X["Tiempo"].apply(lambda x:d[x])
X.head()
X=X.dropna()
etiquetas_true=np.array(X["Tipo"])
etiquetas_true[:5]

# data=np.array(X.drop(["Tipo"], 1))
# # data=np.array(X.drop(["Tipo"], 1))
# y=np.array(X["Tipo"])
# y
# data[:5]

data=X.drop(["Tipo"], 1)
# data=np.array(X.drop(["Tipo"], 1))
data=np.array(data.drop(["BrigN"], 1)) #ó BrigN
y=np.array(X["Tipo"])
y
data[:5]

# plt.plot(data[:,0],data[:,1],'k.')
# plt.show()

from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
Tipo_encoded=le.fit_transform(y)
TY=np.array([Tipo_encoded, y])
print(TY)

from sklearn.cluster import KMeans
n = 2
k_means = KMeans(n_clusters=n)
k_means.fit(data)

centroides = k_means.cluster_centers_
etiquetas = k_means.labels_
etiquetas
centroides

plt.plot(data[etiquetas==0,0],data[etiquetas==0,1],'b.', label='cluster 1')
plt.plot(data[etiquetas==1,0],data[etiquetas==1,1],'r.', label='cluster 2')
# plt.plot(data[etiquetas==2,0],data[etiquetas==2,1],'g.', label='cluster 3')
# plt.plot(data[etiquetas==3,0],data[etiquetas==3,1],'y.', label='cluster 4')
# plt.plot(data[etiquetas==4,0],data[etiquetas==4,1],'k.', label='cluster 5')
# plt.plot(data[etiquetas==5,0],data[etiquetas==5,1],'m.', label='cluster 6')
# plt.plot(data[etiquetas==6,0],data[etiquetas==6,1],'c.', label='cluster 7')
# plt.plot(data[etiquetas==7,0],data[etiquetas==7,1],'k.', label='cluster 8')
# plt.plot(data[etiquetas==8,0],data[etiquetas==8,1],'r.', label='cluster 9')
# plt.plot(data[etiquetas==9,0],data[etiquetas==9,1],'b.', label='cluster 10')
# #plt.plot(data[etiquetas==10,0],data[etiquetas==10,1],'y.', label='cluster 11')
#plt.plot(data[etiquetas==11,0],data[etiquetas==11,1],'g.', label='cluster 12')

plt.plot(centroides[:,0],centroides[:,1],'k.',markersize=15, label='centroides')

plt.legend(loc='best')
plt.show()

adjusted_rand_score(Tipo_encoded, etiquetas)
adjusted_mutual_info_score(Tipo_encoded, etiquetas)
print(etiquetas_true[:], etiquetas[:])


##TRANSFORMO LOS DATOS COM UMAP Y APLICO KMEANS

fig=plt.figure(figsize=(30,10)) #creamos un canvas o figura de 30 x 10 pix
plt.subplot2grid((2,3),(0,0))

standard_embedding = umap.UMAP(random_state=42).fit_transform(data.data)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c="red", s=10.0, cmap='Spectral');

plt.subplot2grid((2,3),(0,1))
kmeans_labels = cluster.KMeans(n_clusters=n).fit_predict(data.data)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=10.0, cmap='Spectral');

print(adjusted_rand_score(etiquetas_true, kmeans_labels),
adjusted_mutual_info_score(etiquetas_true, kmeans_labels))#Métricas para clustering+UMAP

