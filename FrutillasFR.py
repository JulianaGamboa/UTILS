# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:11:43 2020

@author: Juliana Gamboa
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

frutis=pd.read_csv("Tipo224.csv")

#visualizamos los primeros 5 datos
print(frutis.head())

#Eliminamos la 1ra columna ID
#iris = iris.drop("id", axis=1)
print("Información del dataset: ")
print(frutis.info())
print("Descripción del dataset: ")
print(frutis.describe())
print("Distribución de las muestras: ")
print(frutis.groupby("Tipo").size())

#import matplotlib.pyplot as plt #no puedo graficar porque x parece no ser numérico?
###Gráficas que faltan (ejercicio)
#fig=frutis[frutis.Tiempo=="Largo"].plot(kind="scatter", x="%Ret_Brillo_Norm", y="%Ret_Area", color="red", label="Largo")
#frutis[frutis.Tiempo=="Medio"].plot(kind="scatter", x="%Ret_Brillo_Norm", y="%Ret_Area", color="yellow", label="Medio", ax=fig)
#frutis[frutis.Tiempo=="Corto"].plot(kind="scatter", x="%Ret_Brillo_Norm", y="%Ret_Area", color="green", label="Corto", ax=fig)
##
#fig.set_xlabel("%Ret_Brillo_Norm")
#fig.set_ylabel("%Ret_Area")
#fig.set_title("Brillo vs Area")
#plt.show()

####APLICACIÓN DE ALGORITMOS DE MACHINE LEARNING####
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix


###MODELO CON TODOS LOS DATOS###
#Separo los datos (x) de las etiquetas (y)
X = np.array(frutis.drop(["Tipo"], 1))
y = np.array(frutis["Tipo"])

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
Tipo_encoded=le.fit_transform(y)
print(Tipo_encoded)
#print(y)
#Separo los datos de train (enetrenamiento) y test (prueba)
X_train, X_test, y_train, y_test = train_test_split(X, Tipo_encoded, test_size=0.2)
print("Son ", len(X_train), " datos de entrenamiento y ", len(X_test), " datos de prueba")

###APLICO MODELOS DE MACHINE LEARNING###

##REGRESIÓN LOGÍSTICA##
algoritmo = LogisticRegression(solver='lbfgs', multi_class='auto')
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
plot_confusion_matrix(algoritmo, X_train, y_train)
print("Precision de Regresión Logística (train): ", algoritmo.score(X_train, y_train))
print("Precision de Regresión Logística (test): ", algoritmo.score(X_test, y_test))
#print ("{0:.15f}".format(algoritmo.score(X_train, y_train)))


##ALGORITMO DE MÁQUINAS DE VECTORES DE SOPORTE##
algoritmo = SVC (gamma='scale')
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
plot_confusion_matrix(algoritmo, X_train, y_train)
print("Precision de algoritmo SVC (train): ", algoritmo.score(X_train, y_train))
print("Precision de algoritmo SVC (test): ", algoritmo.score(X_test, y_test))
#print ("{0:.15f}".format(algoritmo.score(X_train, y_train)))

##ALGORITMO DE VECINOS MÁS CERCAMOS##
algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
plot_confusion_matrix(algoritmo, X_train, y_train)
print("Precision de algoritmo KNeighbors (train): ", algoritmo.score(X_train, y_train))
print("Precision de algoritmo KNeighbors (test): ", algoritmo.score(X_test, y_test))
#print ("{0:.15f}".format(algoritmo.score(X_train, y_train)))

##ALGORITMO DE ÁRBOLES DE DECISIÓN##
from sklearn import tree
import graphviz
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
plot_confusion_matrix(algoritmo, X_test, y_test)
fruti_data = tree.export_graphviz(algoritmo, out_file=None)
graph = graphviz.Source(fruti_data)
graph.render("Frutillas") #se guarda con este nombre como pdf
frutis_data = tree.export_graphviz(algoritmo, out_file=None, 
                                feature_names= None,  
                                class_names= None,  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(frutis_data)
graph

# tree.plot_tree(algoritmo) 
print("Precision de algoritmo Decision Tree (train): ", algoritmo.score(X_train, y_train))
print("Precision de algoritmo Decision Tree (test): ", algoritmo.score(X_test, y_test))
#print ("{0:.15f}".format(algoritmo.score(X_train, y_train)))
#print(X_test, Y_pred)