import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

frutis=pd.read_csv("Data/1.DataFR_OD_T30.csv")

#visualizamos los primeros 5 datos
print(frutis.head())

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

###MODELO CON TODOS LOS DATOS### random_state=NONE
#Separo los datos (x) de las etiquetas (y)
X = np.array(frutis.drop(["Tipo"], 1))
y = np.array(frutis["Tipo"])

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
Tipo_encoded=le.fit_transform(y)
TY=np.array([Tipo_encoded, y])
print(TY)
#Separo los datos de train (entrenamiento) y test (prueba)
X_train, X_test, y_train, y_test = train_test_split(X, Tipo_encoded, test_size=0.4) #después de test_size podría incluir random_state=seed_value (si el seed_value = 10, el score será siempre el mismo). Puedo poner un value y chequear con el loop de más adelante :)
print("Son ", len(X_train), " datos de entrenamiento y ", len(X_test), " datos de prueba")

###APLICO MODELOS DE MACHINE LEARNING###

##ALGORITMO DE ÁRBOLES DE DECISIÓN##

algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
plot_confusion_matrix(algoritmo, X_test, y_test)
print("Precision de algoritmo Decision Tree (train): ", algoritmo.score(X_train, y_train))
print("Precision de algoritmo Decision Tree (test): ", algoritmo.score(X_test, y_test))


###MODELO CON TODOS LOS DATOS### random_state=seed
model=DecisionTreeClassifier()
results_DT = []
def test_seed(seed):
    X_train, X_test, y_train, y_test = train_test_split(
    X, Tipo_encoded,
    test_size=0.4, random_state=seed)
    Y_pred=algoritmo.fit(X_train, y_train).predict(X_test)
    score = algoritmo.score(X_test, y_test)
    results_DT.append([score, seed])
for i in range(100):
    test_seed(i)    
results_DT

results_sorted = sorted(results_DT, key=lambda x: x[0], reverse=False)
print(results_sorted[0])
print(results_sorted[-1])

from sklearn.model_selection import cross_val_score
#cross_val_score

modelo_ols = DecisionTreeClassifier()
X = X
y = Tipo_encoded

results_cross_val_DT = cross_val_score(
    estimator=modelo_ols, 
    X=X,
    y=y,
    scoring="neg_mean_squared_error", 
    cv=50 #cross-validation generator, if None --> 5-fold cross validation
)

results_cross_val_DT 

def rmse_cross_val(estimator, X, y):
    y_pred = estimator.predict(X)
    return np.sqrt(metrics.mean_squared_error(y, y_pred))

resultados_cv_DT = []
for i in range(2,13):
    cv_rmse = cross_val_score(
        estimator=modelo_ols, 
        X=X,
        y=y,
        scoring=rmse_cross_val, 
        cv=i ##pruebo diferentes estrategias de cross-validation 2 a 50 :)
    ).mean()
    resultados_cv_DT.append(cv_rmse)

resultados_cv_DT

import matplotlib.pyplot as plt
plt.plot(resultados_cv_DT)

from sklearn.model_selection import cross_validate
scoring = {"mae": "neg_mean_absolute_error", "rmse": rmse_cross_val}
estimator = modelo_ols
scores_DT = cross_validate(estimator, X,
                        y, scoring=scoring,
                         cv=10, return_train_score=True)
pd.DataFrame(scores_DT).mean()
