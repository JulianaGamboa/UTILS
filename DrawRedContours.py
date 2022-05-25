# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:59:05 2020

@author: Jule
"""
import cv2 as cv
import numpy as np 
  
# Let's load a simple image with 3 black squares 
image= cv.imread("t10 (7).jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
  
# Find Canny edges 
edged = cv.Canny(gray, 30, 200) 
#cv2.waitKey(0) 
  
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours = cv.findContours(edged,  
    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
  
cv.imshow('Canny Edges After Contouring', edged) 

#cv2.waitKey(0) 
  
print("Number of Contours found = " + str(len(contours))) 
  
##Dibujo el rectángulo que encierra la frutilla (sin rotación)##

cnt=contours[0]
x,y,w,h = cv.boundingRect(cnt)
rectangle=cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) #Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
cv.imshow("REC", rectangle)
print(w,h)

##Dibujo el rectángulo que encierra la frutilla (con rotación)##
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl

cnt=contours[0]
x, y, w, h = cv.boundingRect(cnt)
rect1 = cv.rectangle(image.copy(),(x,y),(x+w,y+h),(0,255,0),3) # not copying here will throw an error
print("x:{0}, y:{1}, width:{2}, height:{3}".format(x, y, w, h))
plt.imshow(rect1)
plt.show()

_,contours,_ = cv.findContours(edged.copy(), 1, 1) # not copying here will throw an error
rect = cv.minAreaRect(contours[0]) # basically you can feed this rect into your classifier
(x,y),(w,h), a = rect # a - angle
print("x:{0}, y:{1}, width:{2}, height:{3}".format(x, y, w, h))

box = cv.boxPoints(rect)
box = np.int0(box) #turn into ints
rect2 = cv.drawContours(image.copy(),[box],0,(0,0,255),3)

plt.imshow(rect2)
plt.show()


