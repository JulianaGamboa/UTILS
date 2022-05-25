# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:47:10 2020

@author: July Gamboa
"""
"""ESTO ES LO QUE DEFINO COMO AREA HUE EN PAPER ML"""

# Umbral de rojo
# Elegimos el umbral de rojo en HSV
import numpy as np
import cv2
from matplotlib import pyplot as plt

imagen = cv2.imread('P201050-1.png') #'t100 (6).jpg'
#plt.imshow(imagen)
img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
umbral_bajo1 = (160,0,60) #(160,0,1), Sat: (160,100,0)
umbral_alto1 = (179,255,255)
# Elegimos el segundo umbral de rojo en HSV
umbral_bajo2 = (0,0,60) #(0,0,1)
umbral_alto2 = (20,255,255)
# hacemos la mask y filtramos en la original
mask1 = cv2.inRange(img_hsv, umbral_bajo1, umbral_alto1)
mask2 = cv2.inRange(img_hsv, umbral_bajo2, umbral_alto2)
mask = mask1 + mask2
res = cv2.bitwise_and(imagen, imagen, mask=mask)
# imprimimos los resultados
plt.subplot(1, 2, 1)
plt.imshow(mask)
plt.subplot(1, 2, 2)
plt.imshow(res)
plt.show()

array=np.asarray(mask)
array=array/255
print(array)
arrays=[]
unique, counts = np.unique(array, return_counts=True)
dict((unique, counts))
counts[1]