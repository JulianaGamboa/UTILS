# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:08:00 2021

@author: Juli Gamboa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
# from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

X =pd.read_csv("Dataset_completo.csv")
X.head()
X=X.dropna()

"###################################################"
"Figure 4. SSE Values for FR REDUCED DATASET (n:490)"
"###################################################"

"###CONFIGURACIÓN C5###"
"""ARMAR CONFIGURACIONES"""
"""C1 to C9 ver trabajo HEFAT: 
   C1: short<20 , 30<intermediate<60 , 70<long
   C2: short<30 , 40<intermediate<60 , 70<long
   C3: short<40 , 50<intermediate<60 , 70<long
   C4: short<30 , 40<intermediate<70 , 80<long
   #C5: short<40 , 50<intermediate<70 , 80<long
   C6: short<50 , 60<intermediate<70 , 80<long
   C7: short<40 , 50<intermediate<80 , 90<long
   C8: short<50 , 60<intermediate<80 , 90<long
   C9: short<60 , 70<intermediate<80 , 90<long
       """
"""MÁSCARAS DE CONFIGURACIONES + CLUSTERING (kmeans & agglomerative clustering)"""
"""C1_SHORT"""
C1_mask_short=(X['Time']<=20) & (X['Tipo']=="OD") ##Voy cambiando los valores en función de las configuraciones  
FR_filtered_C1_SHORT = X[C1_mask_short]
FR_filtered_C1_SHORT.describe()
FR_filtered_C1_SHORT.head()
FR_filtered_C1_SHORT=FR_filtered_C1_SHORT.assign(Time_category="Short")
print(FR_filtered_C1_SHORT)

"""C1_INTERMEDIATE"""
C1_mask_interm=(X['Time']>=30) & (X['Time']<=60) & (X["Tipo"]=="OD") ##Voy cambiando los valores en función de las configuraciones  
FR_filtered_C1_INTERMEDIATE = X[C1_mask_interm]
FR_filtered_C1_INTERMEDIATE.describe()
FR_filtered_C1_INTERMEDIATE.head()
FR_filtered_C1_INTERMEDIATE=FR_filtered_C1_INTERMEDIATE.assign(Time_category="Intermediate")
print(FR_filtered_C1_INTERMEDIATE)

"""C1_LONG"""
C1_mask_long= (X['Time']>=70) & (X["Tipo"]=="OD")  ##Voy cambiando los valores en función de las configuraciones
FR_filtered_C1_LONG = X[C1_mask_long]
FR_filtered_C1_LONG.describe()
FR_filtered_C1_LONG.head()
FR_filtered_C1_LONG=FR_filtered_C1_LONG.assign(Time_category="Long")
print(FR_filtered_C1_LONG)

frames = [FR_filtered_C1_SHORT, FR_filtered_C1_INTERMEDIATE, FR_filtered_C1_LONG]
FR_filtered_C1_ = pd.concat(frames, keys=['Short', 'Intermediate', "Long"])
FR_filtered_C1_.head()

true_labels=np.array(FR_filtered_C1_["Time_category"]) #ó Tipo
true_labels[:5]
features=FR_filtered_C1_.drop(["Time_category"], 1)
features=features.drop(["Tipo"], 1)
features=features.drop(["Time"], 1)
features=features.drop(["Tiempo"], 1)
features=features.drop(["Plate"], 1)
features=features.drop(["Coating"], 1)
features[:5]

"""###SSE PARA DATASET COMPLETO###"""

features= ["RetAreaHue", "RetBrig", "RetSat", "RetAreaRec", "RetWE", "RetHE", "RetRA"]
x = X.loc[:, features].values

scaler = StandardScaler()
scaled_features = scaler.fit_transform(x)
scaled_features[:5]
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)
    
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse, color='#ff5500')
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)

kl.elbow

"##########################################################"
"Results Table 2: ir variando las configuraciones de arriba"
"##########################################################"

kmeans = KMeans(n_clusters=3, **kmeans_kwargs)
kmeans.fit(scaled_features)
kmeans.labels_.shape

ari_C5_ = round(adjusted_rand_score(true_labels, kmeans.labels_), 2)
ami_C5_ = round(adjusted_mutual_info_score(true_labels, kmeans.labels_), 2)
print("For C5 dataset the results of kmeans clustering are: ")
print("ari_: ", ari_C5_)
print("ami_: ", ami_C5_)

"##################################################################################################################################################"
"Results Table 3: K-means scores obtained for sample type (FR and OD) clustering at different drying time categories (c5).Complete dataset, n: 1150"
"##################################################################################################################################################"
"""ARMAR CONFIGURACIONES"""
"""C1 to C9 ver trabajo HEFAT: 
   C1: short<20 , 30<intermediate<60 , 70<long
   C2: short<30 , 40<intermediate<60 , 70<long
   C3: short<40 , 50<intermediate<60 , 70<long
   C4: short<30 , 40<intermediate<70 , 80<long
   #C5: short<40 , 50<intermediate<70 , 80<long
   C6: short<50 , 60<intermediate<70 , 80<long
   C7: short<40 , 50<intermediate<80 , 90<long
   C8: short<50 , 60<intermediate<80 , 90<long
   C9: short<60 , 70<intermediate<80 , 90<long
       """
X.columns
X=X.loc[:,["Tipo","Time", "RetAreaHue", "RetBrig", "RetSat", "RetAreaRec", "RetWE", "RetHE", "RetRA"]]
#X=X.loc[:,["Tipo","Time", "AreaN", "SatN"]]
X.columns

X=X[X["Tipo"]=="OD"]
#X=X[X["Tipo"]=="FR"]

"""MÁSCARAS DE CONFIGURACIONES + CLUSTERING (kmeans & agglomerative clustering)"""
"""C1_SHORT"""
C1_mask_short=X['Time']<=20  
filtered_C1_SHORT = X[C1_mask_short]
filtered_C1_SHORT.describe()
true_labels=np.array(filtered_C1_SHORT["Tipo"]) #ó Tiempo
true_labels[:5]
features=filtered_C1_SHORT.drop(["Tipo"], 1)
features=features.drop(["Time"], 1)
features[:5]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
kmeans.fit(scaled_features)
ari_C1_short = adjusted_rand_score(true_labels, kmeans.labels_)
ami_C1_short = adjusted_mutual_info_score(true_labels, kmeans.labels_)
score_C1_short = silhouette_score(scaled_features, kmeans.labels_)
print("For short times dataset: ")
print("silhouette_score: ", score_C1_short)
print("ari_short: ", ari_C1_short)
print("ami_short: ", ami_C1_short)

"""C1_INTERMEDIATE"""
C1_mask_interm=(X['Time']>=30) & (X['Time']<=60)  
filtered_C1_INTERMEDIATE = X[C1_mask_interm]
filtered_C1_INTERMEDIATE.describe()
true_labels=np.array(filtered_C1_INTERMEDIATE["Tipo"]) #ó Tiempo
true_labels[:5]
features=filtered_C1_INTERMEDIATE.drop(["Tipo"], 1)
features=features.drop(["Time"], 1)
y=np.array(filtered_C1_INTERMEDIATE["Tipo"])
y
features[:5]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
kmeans.fit(scaled_features)
score_C1_interm = silhouette_score(scaled_features, kmeans.labels_)
ari_C1_interm = adjusted_rand_score(true_labels, kmeans.labels_)
ami_C1_interm = adjusted_mutual_info_score(true_labels, kmeans.labels_)
print("For intermediate times dataset: ")
print("silhouette_score: ", score_C1_interm)
print("ari_interm: ", ari_C1_interm)
print("ami_interm: ", ami_C1_interm)

"""C1_LONG"""
C1_mask_long=X['Time']>=70  
filtered_C1_LONG = X[C1_mask_long]
filtered_C1_LONG.describe()
true_labels=np.array(filtered_C1_LONG["Tipo"]) #ó Tiempo
true_labels[:5]
features=filtered_C1_LONG.drop(["Tipo"], 1)
features=features.drop(["Time"], 1)
y=np.array(filtered_C1_LONG["Tipo"])
y
features[:5]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
kmeans.fit(scaled_features)
score_C1_long = silhouette_score(scaled_features, kmeans.labels_)
ari_C1_long = adjusted_rand_score(true_labels, kmeans.labels_)
ami_C1_long = adjusted_mutual_info_score(true_labels, kmeans.labels_)
print("For long times dataset: ")
print("silhouette_score: ", score_C1_long)
print("ari_long: ", ari_C1_long)
print("ami_long: ", ami_C1_long)

###Grafico Figura de clusters by KMEANS CLUSTERS
centroides = kmeans.cluster_centers_
etiquetas = kmeans.labels_
etiquetas.shape
centroides

plt.plot(scaled_features[etiquetas==1,0],scaled_features[etiquetas==1,1], "b.", color='#00aaff', label='OD cluster')
plt.plot(scaled_features[etiquetas==0,0],scaled_features[etiquetas==0,1], "r.", color='#ff5500', label='FR cluster')

plt.plot(centroides[:,0],centroides[:,1],'k.',markersize=15, label='centroids')

plt.legend(loc='best')
plt.show()

###TRUE LABELS
centroides = kmeans.cluster_centers_
true_etiquetas = true_labels
true_etiquetas.shape
centroides

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_etiquetas_encoded=le.fit_transform(true_etiquetas)
print(True_etiquetas_encoded)
True_etiquetas_encoded.shape

plt.plot(scaled_features[True_etiquetas_encoded==1,0],scaled_features[True_etiquetas_encoded==1,1], "b.", color='#00aaff', label='OD true label')
plt.plot(scaled_features[True_etiquetas_encoded==0,0],scaled_features[True_etiquetas_encoded==0,1], "r.", color='#ff5500', label='FR true label')

#plt.plot(centroides[:,0],centroides[:,1],'k.',markersize=15, label='centroides')

plt.legend(loc='best')
plt.show()

"############################################################################################################################################################"
"Results Table 3: K-means scores obtained for sample type (FR and OD) clustering at different drying time categories (c5). Reduced dataset: nonCoated, n: 650"
"############################################################################################################################################################"
"""ARMAR CONFIGURACIONES"""
"""C1 to C9 ver trabajo HEFAT: 
   C1: short<20 , 30<intermediate<60 , 70<long
   C2: short<30 , 40<intermediate<60 , 70<long
   C3: short<40 , 50<intermediate<60 , 70<long
   C4: short<30 , 40<intermediate<70 , 80<long
   #C5: short<40 , 50<intermediate<70 , 80<long
   C6: short<50 , 60<intermediate<70 , 80<long
   C7: short<40 , 50<intermediate<80 , 90<long
   C8: short<50 , 60<intermediate<80 , 90<long
   C9: short<60 , 70<intermediate<80 , 90<long
       """
X.columns
X=X.loc[:,["Tipo","Time", "Coating", "AreaN", "BrigN", "SatN", "AR", "WE", "HE", "RAD"]]
#X=X.loc[:,["Tipo","Time", "Coating", "AreaN", "SatN"]]
X.columns
## REDUCED DATASET
RD=X["Coating"]=="noncoated" ##Escenario 1
#RD=X["Coating"]=="coated"   ##Escenario 2
X_RD=X[RD]
X_RD.head()
X_RD.describe()

"""MÁSCARAS DE CONFIGURACIONES + CLUSTERING (kmeans & agglomerative clustering)"""
"""C1_SHORT"""
C1_mask_short=X_RD['Time']<=40  
filtered_C1_SHORT = X_RD[C1_mask_short]
filtered_C1_SHORT.describe()
true_labels=np.array(filtered_C1_SHORT["Tipo"]) #ó Tiempo
true_labels[:5]
features=filtered_C1_SHORT.drop(["Tipo"], 1)
features=features.drop(["Time"], 1)
features=features.drop(["Coating"], 1)
features[:5]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
kmeans.fit(scaled_features)
ari_C1_short = adjusted_rand_score(true_labels, kmeans.labels_)
ami_C1_short = adjusted_mutual_info_score(true_labels, kmeans.labels_)
score_C1_short = silhouette_score(scaled_features, kmeans.labels_)
print("For short times dataset: ")
print("silhouette_score: ", score_C1_short)
print("ari_short: ", ari_C1_short)
print("ami_short: ", ami_C1_short)

"""C1_INTERMEDIATE"""
C1_mask_interm=(X_RD['Time']>=50) & (X_RD['Time']<=70)  
filtered_C1_INTERMEDIATE = X_RD[C1_mask_interm]
filtered_C1_INTERMEDIATE.describe()
true_labels=np.array(filtered_C1_INTERMEDIATE["Tipo"]) #ó Tiempo
true_labels[:5]
features=filtered_C1_INTERMEDIATE.drop(["Tipo"], 1)
features=features.drop(["Time"], 1)
features=features.drop(["Coating"], 1)
features[:5]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
kmeans.fit(scaled_features)
score_C1_interm = silhouette_score(scaled_features, kmeans.labels_)
ari_C1_interm = adjusted_rand_score(true_labels, kmeans.labels_)
ami_C1_interm = adjusted_mutual_info_score(true_labels, kmeans.labels_)
print("For intermediate times dataset: ")
print("silhouette_score: ", score_C1_interm)
print("ari_interm: ", ari_C1_interm)
print("ami_interm: ", ami_C1_interm)

###k-means Labels
centroides = kmeans.cluster_centers_
etiquetas = kmeans.labels_
etiquetas
centroides

plt.plot(scaled_features[etiquetas==0,0],scaled_features[etiquetas==0,1],'b.', label='cluster 1')
plt.plot(scaled_features[etiquetas==1,0],scaled_features[etiquetas==1,1],'r.', label='cluster 2')

plt.plot(centroides[:,0],centroides[:,1],'k.',markersize=15, label='centroides')

plt.legend(loc='best')

###TRUE LABELS
centroides = kmeans.cluster_centers_
true_etiquetas = true_labels
true_etiquetas.shape
centroides

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_etiquetas_encoded=le.fit_transform(true_etiquetas)
print(True_etiquetas_encoded)
True_etiquetas_encoded.shape

plt.plot(scaled_features[True_etiquetas_encoded==1,0],scaled_features[True_etiquetas_encoded==1,1], "b.", color='#00aaff', label='OD true label')
plt.plot(scaled_features[True_etiquetas_encoded==0,0],scaled_features[True_etiquetas_encoded==0,1], "r.", color='#ff5500', label='FR true label')

#plt.plot(centroides[:,0],centroides[:,1],'k.',markersize=15, label='centroides')

plt.legend(loc='best')
plt.show()
plt.show()

"""C1_LONG"""
C1_mask_long=X_RD['Time']>=80  
filtered_C1_LONG = X_RD[C1_mask_long]
filtered_C1_LONG.describe()
true_labels=np.array(filtered_C1_LONG["Tipo"]) #ó Tiempo
true_labels[:5]
features=filtered_C1_LONG.drop(["Tipo"], 1)
features=features.drop(["Time"], 1)
features=features.drop(["Coating"], 1)
features[:5]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
kmeans.fit(scaled_features)
score_C1_long = silhouette_score(scaled_features, kmeans.labels_)
ari_C1_long = adjusted_rand_score(true_labels, kmeans.labels_)
ami_C1_long = adjusted_mutual_info_score(true_labels, kmeans.labels_)
print("For long times dataset: ")
print("silhouette_score: ", score_C1_long)
print("ari_long: ", ari_C1_long)
print("ami_long: ", ami_C1_long)

###k-means Labels
centroides = kmeans.cluster_centers_
etiquetas = kmeans.labels_
etiquetas
centroides

plt.plot(scaled_features[etiquetas==0,0],scaled_features[etiquetas==0,1], "b.", color='#00aaff', label='OD cluster')
plt.plot(scaled_features[etiquetas==1,0],scaled_features[etiquetas==1,1], "r.", color='#ff5500', label='FR cluster')

plt.plot(centroides[:,0],centroides[:,1],'k.',markersize=15, label='centroids')

plt.legend(loc='best')
plt.show()

###TRUE LABELS
centroides = kmeans.cluster_centers_
true_etiquetas = true_labels
true_etiquetas.shape
centroides

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
True_etiquetas_encoded=le.fit_transform(true_etiquetas)
print(True_etiquetas_encoded)
True_etiquetas_encoded.shape

plt.plot(scaled_features[True_etiquetas_encoded==1,0],scaled_features[True_etiquetas_encoded==1,1], "b.", color='#00aaff', label='OD true label')
plt.plot(scaled_features[True_etiquetas_encoded==0,0],scaled_features[True_etiquetas_encoded==0,1], "r.", color='#ff5500', label='FR true label')

#plt.plot(centroides[:,0],centroides[:,1],'k.',markersize=15, label='centroides')

plt.legend(loc='best')
plt.show()
plt.show()