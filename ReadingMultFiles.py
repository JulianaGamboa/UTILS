
# import os
# frut_dir = "./FR/Positions"

# import pandas as pd
# stats = pd.DataFrame(columns = ("position", "sample"))
# sample_num = 1
# for position in os.listdir(frut_dir):
#     print(position)
#     for time in os.listdir(frut_dir + "/" + position):
#         #print (sample)
#         #for fru in os.listdir(book_dir + "/" + position + "/" + sample):
#         inputfile = frut_dir + "/" + position + "/" + time
#         print(inputfile)
#         # text = read_book(inputfile)
#         # (num_unique, counts) = word_stats(count_word_fast(text))
#         stats.loc[sample_num] = position, time.replace(".jpg", "")
#         sample_num +=1    
# stats

##############################
##Cálculo de áreas x carpeta##
##############################

import os
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt

# frut_dir = "./FR/Positions"
# def read_img (inputfile):
#     imagen = cv2.imread(inputfile)
#     return imagen
#imagen=read_img("t0 (1).jpg")
    #plt.imshow(imagen)
def cont_HSV (im):  
    img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    umbral_bajo1 = (160,0,1)
    umbral_alto1 = (179,255,255)
    # Elegimos el segundo umbral de rojo en HSV
    umbral_bajo2 = (0,0,1)
    umbral_alto2 = (20,255,255)
    # hacemos la mask y filtramos en la original
    mask1 = cv2.inRange(img_hsv, umbral_bajo1, umbral_alto1)
    mask2 = cv2.inRange(img_hsv, umbral_bajo2, umbral_alto2)
    mask = mask1 + mask2
    res = cv2.bitwise_and(imagen, imagen, mask=mask)
    array=np.asarray(mask)
    array=array/255
    #print(array)
    arrays=[]
    unique, counts = np.unique(array, return_counts=True)
    #return dict((unique, counts))
    return [counts]
#cont_HSV (imagen)

Area=[]
stats = pd.DataFrame(columns = ("position", "sample", "area"))
sample_num = 1
for position in os.listdir(frut_dir):
    #print(position)
    for time in os.listdir(frut_dir + "/" + position):
        #print (sample)
        #for fru in os.listdir(book_dir + "/" + position + "/" + sample):
        inputfile = frut_dir + "/" + position + "/" + time
        imagen = cv2.imread(inputfile)
        #plt.imshow(imagen)
        print(inputfile)
        #frut= read_img(inputfile)
        area=cont_HSV(imagen)
        Area.append(area)        
        stats.loc[sample_num] = position, time.replace(".jpg", ""), area
        sample_num +=1     
stats.head()

