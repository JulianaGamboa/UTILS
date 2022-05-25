### PRÁCTICA DE PANDAS ####
###########################
import pandas as pd

x=pd.Series ([6,3,8,6], index=["q", "w", "e", "r"])
y=pd.Series ([4,5,7,8], index=["r", "t", "q", "p"])
x
y

x+y #suma los valores de los índices que se repiten en las dos series si no pone NaN

x.index
sorted(x.index) #ordena los índices alfabéticamente
x.reindex(sorted(x.index)) #como salida da los valores numéricos ordenados (8,6,6,3)


x[["r", "w"]]
age = {"Tim":29, "Jim":31, "Pam":27, "Sam":35}
x = pd.Series(age)
x

data= {"name" : ["Tim", "Jim", "Pam", "Sam"],
       "age": [29, 31, 27, 35],
       "ZIP": ["1900", "1150", "1250", "0089"]}

x = pd.DataFrame(data, columns=["name", "age", "ZIP"])

x["name"] #de las dos maneras (esta y la de abajo) accedo a la columna name
x.name

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

