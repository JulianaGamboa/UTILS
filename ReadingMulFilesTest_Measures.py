"""#########################################################
###############CALCULO DE MEDIDAS BOX ######################
#########################################################"""

"""Este es el código para leer todas las imágenes de una carpeta y sus subcarpetas de posición!!!
Devuelve un dataFrame con 7 columnas: position, sample (posición), area [box], w [box], h [box], radius [box].
Ver Measures.py para entender el código para 1 foto."""

##########################################
##Cálculo de medidas frutillas x carpeta##
##########################################

import os
import numpy as np
import cv2
import pandas as pd
# from matplotlib import pyplot as plt

# sample_num = 1
img = cv2.pyrDown(cv2.imread("./FRAMBUESAS FRESCAS/Pre_OD/P7u.png", cv2.IMREAD_UNCHANGED))

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 0, 255, cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
# find bounding box coordinates
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
print(w,h)
# find minimum area
rect = cv2.minAreaRect(c)
# calculate coordinates of the minimum area rectangle
box = cv2.boxPoints(rect)
# normalize coordinates to integers
box = np.int0(box) 
# draw contours
cv2.drawContours(img, [box], 0, (0,0, 255), 3)
# calculate center and radius of minimum enclosing circle
(x,y),radius = cv2.minEnclosingCircle(c)
# cast to integers
center = (int(x),int(y))
radius = int(radius)
# draw the circle
img = cv2.circle(img,center,radius,(0,255,0),2)
cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow("contours", img)

ARect=[]
Weight=[]
Height=[]
Radius=[]
Arect=w*h
ARect.append(Arect)
Weight.append(w)
Height.append(h)
Radius.append(radius)
# stats=pd.DataFrame(columns=("ARect", "Weight", "Height"))
# stats.loc[sample_num]=ARect, Weight, Height
# sample_num += 1

#########################################
import os
import numpy as np
import cv2
import pandas as pd

frut_dir = "./FRAMBUESAS FRESCAS"
ARect=[]
Weight=[]
Height=[]
Radius=[]
stats=pd.DataFrame(columns = ("position", "sample", "weight", "height", "ARect", "Radius"))
sample_num = 0
for position in os.listdir(frut_dir):
    #print(position)
    for time in os.listdir(frut_dir + "/" + position):
        #print (sample)
        #for fru in os.listdir(book_dir + "/" + position + "/" + sample):
        inputfile = frut_dir + "/" + position + "/" + time
        img = cv2.imread(inputfile)
        #plt.imshow(imagen)
        print(inputfile)
        #frut= read_img(inputfile)
        ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 0, 255, cv2.THRESH_BINARY)
        image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            (x,y),radius = cv2.minEnclosingCircle(c)
            # print(w,h)
            if w>10:
                # find minimum area
                rect = cv2.minAreaRect(c)
                # calculate coordinates of the minimum area rectangle
                box = cv2.boxPoints(rect)
                # normalize coordinates to integers
                box = np.int0(box)  
                A=w*h
                ARect.append(A)
                Weight.append(w)
                Height.append(h)
            if radius>10:
            # # # cast to integers
                center = (int(x),int(y))
                radius = int(radius)
                Radius.append(radius)
            # find minimum area
            # rect = cv2.minAreaRect(c)
            # ## calculate coordinates of the minimum area rectangle
            # box = cv2.boxPoints(rect)
            # ## normalize coordinates to integers
            # box = np.int0(box)
            stats.loc[sample_num] = position, time.replace(".png", ""), Weight, Height, ARect, Radius
        sample_num +=1     
stats.head()

W=[]
sample_num=0
for e in Weight:
    # e=max(e)
    a=np.array(e)
    W.append(a)
    sample_num +=1
H=[]
sample_num=0
for f in Height:
    # e=max(e)
    b=np.array(f)
    H.append(b)
    sample_num +=1
AR=[]
sample_num=0
for g in ARect:
    # e=max(e)
    d=np.array(g)
    AR.append(d)
    sample_num +=1
RA=[]
sample_num=0
for n in Radius:
    # e=max(e)
    j=np.array(n)
    RA.append(j)
    sample_num +=1
    
We=np.vstack(W) #convierte los valores [[]] en dos columnas de un data frame :)
#S[1,1]/(S[1,0]+S[1,1])
We_DF=pd.DataFrame(We)
We_DF.head() #los vuelvo a convertir a DataFrame para incluirlos en stats
We_DF

He=np.vstack(H) #convierte los valores [[]] en dos columnas de un data frame :)
#S[1,1]/(S[1,0]+S[1,1])
He_DF=pd.DataFrame(He)
He_DF.head() #los vuelvo a convertir a DataFrame para incluirlos en stats
He_DF

AR=np.vstack(ARect) #convierte los valores [[]] en dos columnas de un data frame :)
#S[1,1]/(S[1,0]+S[1,1])
AR_DF=pd.DataFrame(AR)
AR_DF.head() #los vuelvo a convertir a DataFrame para incluirlos en stats
AR_DF

RA=np.vstack(Radius) #convierte los valores [[]] en dos columnas de un data frame :)
#S[1,1]/(S[1,0]+S[1,1])
RA_DF=pd.DataFrame(RA)
RA_DF.head() #los vuelvo a convertir a DataFrame para incluirlos en stats
RA_DF

stats[['Weight']] = We_DF
stats.head()
stats[['Height']] = He_DF
stats.head()
stats[['ARect']] = AR_DF
stats.head()
stats[['Radius']] = RA_DF
stats.head()

stats.to_excel("Measures_crudos_We&He&AR&RA_FRAMBUESAS_FRESCAS.xlsx")
