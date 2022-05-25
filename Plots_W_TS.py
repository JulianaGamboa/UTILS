# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:28:34 2020

@author: Diez
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

frutis=pd.read_csv("OD_FR.csv")

#visualizamos los primeros 5 datos
print(frutis.head())

print("Información del dataset: ")
print(frutis.info())
print("Descripción del dataset: ")
print(frutis.describe())
# print("Distribución de las muestras: ")
# print(frutis.groupby("Time").size())

Y = np.array(frutis.drop(["Time"], 1))
x = np.array(frutis["Time"])
y0=np.mean(np.array(Y[0,:]))
y1=np.mean(np.array(Y[1,:]))
y2=np.mean(np.array(Y[2,:]))
y3=np.mean(np.array(Y[3,:]))
y4=np.mean(np.array(Y[4,:]))
y5=np.mean(np.array(Y[5,:]))
y6=np.mean(np.array(Y[6,:]))
y7=np.mean(np.array(Y[7,:]))
y8=np.mean(np.array(Y[8,:]))
# y9=np.mean(np.array(Y[9,:]))
# y10=np.mean(np.array(Y[10,:]))
mean=[y0, y1, y2, y3, y4, y5, y6, y7, y8]

z0=np.std(np.array(Y[0,:]))
z1=np.std(np.array(Y[1,:]))
z2=np.std(np.array(Y[2,:]))
z3=np.std(np.array(Y[3,:]))
z4=np.std(np.array(Y[4,:]))
z5=np.std(np.array(Y[5,:]))
z6=np.std(np.array(Y[6,:]))
z7=np.std(np.array(Y[7,:]))
z8=np.std(np.array(Y[8,:]))
# z9=np.std(np.array(Y[9,:]))
# z10=np.std(np.array(Y[10,:]))
std=[z0, z1, z2, z3, z4, z5, z6, z7, z8]
# plt.ion()  # Ponemos el modo interactivo
# plt.scatter(x, y, c="g", marker="o")  # Dibujamos un scatterplot
# # plt.axvline(-0.5, color = 'g')  # Dibujamos una línea vertical verde centrada en x = -0.5
# # plt.axvline(-0.5, color = 'g')  # Dibujamos una línea vertical verde centrada en x = 0.5
# # plt.axhline(-0.5, color = 'g')  # Dibujamos una línea horizontal verde centrada en x = -0.5
# # plt.axhline(-0.5, color = 'g')  # Dibujamos una línea horizontal verde centrada en x = 0.5
# # plt.axvspan(-0.5,0.5, alpha = 0.25)  #  Dibujamos un recuadro azul vertical entre x[-0.5,0.5] con transparencia 0.25
# plt.axhspan(64.45,100, alpha = 0.25)  #  Dibujamos un recuadro azul horizontal entre x[-0.5,0.5] con transparencia 0.25

# from seaborn import lmplot
# lmplot(x, mean, data=frutis, fit_reg=False)
# plt.plot(x, mean, color='r', label='FR')
# # plt.plot(max_deep_list, eval_prec, color='b', label='evaluacion')
# # plt.title('Area Retention')
# plt.legend()
# plt.ylabel('Area Ret')
# plt.xlabel("Time (min)")
# plt.show()


std1=[]
for i in range(len(mean)):
    std1.append(mean[i]+std[i])
print (std1)
std2=[]
for i in range(len(mean)):
	std2.append(mean[i]-std[i])
print (std2)

plt.plot(x, mean, color='r', marker='o', markersize=2,
         label='W (kg/ kg DM)')
plt.fill_between(x, std1, std2, color='y')
# plt.plot(max_deep_list, test_mean, color='b', linestyle='--', 
        # marker='s', markersize=5, label='evaluacion')
# plt.fill_between(max_deep_list, test_mean + test_std, 
                 # test_mean - test_std, alpha=0.15, color='b')
plt.grid()
# plt.legend(loc='center right')
plt.ylabel('W (kg/ kg DM)')
plt.xlabel("Time (min)")
plt.show()
