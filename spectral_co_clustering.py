###SPECTRAL CO-CLUSTERING ####
##############################

#ANTES HAY QUE CORRER CLASSIFYING_WHISKIES.PY
### CLASSIFYING WHISKIES ###
import numpy as np
import pandas as pd

whisky = pd.read_csv("whiskies.txt")
whisky["Region"]=pd.read_csv("Regions.txt") #así nomás le agregué una column
whisky.head()
whisky.tail()

whisky.iloc[0:10, 0:3]
whisky.columns
flavors = whisky.iloc[:, 2:14] #extraemos las columnas de flavors
flavors

### EXPLORING CORRELATIONS ###

corr_flavors = pd.DataFrame.corr(flavors)
print(corr_flavors)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig("corr_flavors.pdf")

corr_whisky = pd.DataFrame.corr(flavors.transpose()) #ahora comparamos los whiskies
print(corr_whisky)

plt.figure(figsize=(10,10))
plt.pcolor(corr_whisky)
plt.colorbar()
plt.savefig("corr_whisky.pdf")

#####FIN DE CLASSIFYING_WHISKIES.PY############

from sklearn.cluster import SpectralCoclustering
model = SpectralCoclustering (n_clusters=6, random_state=0) #elegimos 6 clusters porque hay 6 regiones
model.fit (corr_whisky)

model.rows_ #cada fila es 1/6 clusters y cada columna es un whisky, si es true pertenece al cluster, si es false no pertenece
np.sum(model.rows_, axis=1) #cuenta las columnas (axis=1) que son los whiskies que corresponden a cada cluster (true)
model.row_labels_#esto muestra a qué cluster pertenece cada observación(whisky): la observación 0 corresponde al cluster 2, la 1 al 4 y así...
#Remember: Va de 0 a 5 porque especifiqué 6 clusters 

###Comparing correlation matrices###
whisky["Group"]=pd.Series(model.row_labels_, index=whisky.index) #extract the group labels of the model and append them to the whisky table (3 lines)
whisky=whisky.iloc[np.argsort(model.row_labels_)] 
whisky=whisky.reset_index(drop=True)

correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose()) #recalculate the correlation matrix
correlations = np.array(correlations)

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.colorbar()
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.colorbar()
plt.title("Rearranged")
plt.axis("tight")
plt.savefig("correlations.pdf")

#Vemos cómo se agruparon en clusters en la imagen rearranged






