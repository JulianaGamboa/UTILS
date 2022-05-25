# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:00:57 2020

@author: Juli Gamboa
"""

import numpy as np
import pandas as pd
df=pd.read_csv("DATA_3_MEANS.csv")

from sklearn.preprocessing import StandardScaler
features = ["Polyphenols",	"DPPH",	"Flavonoids",	"Carotenoids",	"FRAP",	"Ascorb.Ac.",	"Taste",	"Texture",	"Sweetness",	"Acidity",	"Residual",	"General", "Delta E", "WLSG"
]
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['Samples']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

print(principalComponents)

# ##Suma de varianza explicada por el n_components elegido
np.cumsum(pca.explained_variance_ratio_)

finalDf = pd.concat([principalDf, df[['Samples']]], axis = 1)

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
plt.rcParams.update({'font.size': 8})
labels = ["TP", "DPPH", "FL", "CA", "FRAP", "AA", "TA", "TE", "SW", "AC", "RE", "GE", "DE", "WLSG" ]
txt = ['Fresh fruit', 'A1 60/20/40', 'A2 120/20/60', "A3 180/40/40", "A4 120/30/40", "A5 60/40/40", "A6 60/30/60", "A7 120/40/60", "A8 120/30/40", "A9 120/40/20",  "A10 180/30/20", "A11 120/30/40", "A12 120/20/20", "A13 180/30/60", "A14 60/30/20", "A15 180/20/40"]
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
    plt.xlabel("PC1 (62.4%)", fontsize = 15)
    plt.ylabel("PC2 (20.7%)", fontsize = 15)
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

# label = finalDf["Samples"]
# txt = finalDf.loc[:, "Samples"].values
txt = ['Fresh fruit', 'A1 60/20/40', 'A2 120/20/60', "A3 180/40/40", "A4 120/30/40", "A5 60/40/40", "A6 60/30/60", "A7 120/40/60", "A8 120/30/40", "A9 120/40/20",  "A10 180/30/20", "A11 120/30/40", "A12 120/20/20", "A13 180/30/60", "A14 60/30/20", "A15 180/20/40"]
fig, ax = plt.subplots(1, figsize=(10, 6))
fig.suptitle('Score plot')
ax.scatter(x, y, color="black")
ax.set_xlabel('PC1 (62.4%)', fontsize = 15)
ax.set_ylabel('PC2 (20.7%)', fontsize = 15)
plt.grid()
for i, txt in enumerate(txt):
    ax.annotate(txt, (x[i], y[i]))
# label = finalDf["Samples"]
# txt = finalDf.loc[:, "Samples"].values

###OPTIMOS###
#############

df2=pd.read_csv("DATA_OP.csv")

from sklearn.preprocessing import StandardScaler
features = ["Polyphenols",	"DPPH",	"Flavonoids",	"Carotenoids",	"FRAP",	"Ascorb.Ac.",	"Taste",	"Texture",	"Sweetness",	"Acidity",	"Residual",	"General", "Delta E", "WLSG"
]
# Separating out the features
x = df2.loc[:, features].values
# Separating out the target
y = df2.loc[:,['Samples']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf2= pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

print(principalComponents)
#Scatter plot (Score plot)

finalDf2 = pd.concat([principalDf2, df2[['Samples']]], axis = 1)
final = pd.concat([finalDf, finalDf2], axis = 0)

PC1 = final["principal component 1"]
x = final.loc[:, "principal component 1"].values
PC2 = final["principal component 2"]
y = final.loc[:, "principal component 2"].values

principalComponents = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2'])
PCs = pd.concat([principalDf, principalComponents], axis = 0)

labels = ["TP", "DPPH", "FL", "CA", "FRAP", "AA", "TA", "TE", "SW", "AC", "RE", "GE", "DE", "WLSG" ]
txt = ['Fresh fruit', 'A1 60/20/40', 'A2 120/20/60', "A3 180/40/40", "A4 120/30/40", "A5 60/40/40", "A6 60/30/60", "A7 120/40/60", "A8 120/30/40", "A9 120/40/20",  "A10 180/30/20", "A11 120/30/40", "A12 120/20/20", "A13 180/30/60", "A14 60/30/20", "A15 180/20/40", "OP. Total", "OP. Antioxodants", "OP. Sensory"]
fig, ax = plt.subplots(1, figsize=(10, 6))
# fig.suptitle('Score plot')
ax.scatter(x, y, color="black")
ax.set_xlabel('PC1 (62.4%)', fontsize = 15)
ax.set_ylabel('PC2 (20.7%)', fontsize = 15)
plt.grid()
for i, txt in enumerate(txt):
    ax.annotate(txt, (x[i], y[i]))
# def myplot(score,coeff,labels= labels):
coeff = np.transpose(pca.components_[0:2, :])
xs = PCs[:,0]
ys = PCs[:,1]
n = coeff.shape[0]
scalex = 1.0/(xs.max() - xs.min())
scaley = 1.0/(ys.max() - ys.min())
plt.scatter(xs * scalex,ys * scaley, color = "black", s=5)
        # plt.scatter(xs*0.1,ys*0.1, color = "black", s=5)
for i in range(n):
    plt.arrow(0, 0, coeff[i,0]*10, coeff[i,1]*10,color = 'gray',alpha = 0.5)
        # if labels is None:
        #     plt.text(coeff[i,0]* 0.85, coeff[i,1] * 0.85, "Var"+str(i+1), color = 'gray', ha = 'center', va = 'center')
        # else:
    plt.text(coeff[i,0]* 0.85, coeff[i,1] * 0.85, labels[i], color = 'black', ha = 'center', va = 'center') 
plt.show()