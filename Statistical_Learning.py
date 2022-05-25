### STATISTICAL LEARNING
### SUPERVISED LEARNING
## 5.1.2. Generating example regression data

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n= 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc=0, scale=1, size=n)

plt.figure()
plt.plot(x, y, "o", ms=5)
xx = np.array([0, 10])
plt.plot(xx, beta_0 + beta_1 * xx)
plt.xlabel("x")
plt.ylabel("y")

##5.1.3. Simple linear regression
##5.1.4. Least square estimation in code

rss = []
slopes = np.arange(-10, 15, 0.01)
for slope in slopes:
    rss.append(np.sum((y - beta_0 - slope * x) **2))
ind_min = np.argmin(rss)
ind_min ##find the location of min value of rss
print ("Estimate for the slope: ", slopes[ind_min])

plt.figure()
plt.plot(slopes, rss)
plt.xlabel("Slope")
plt.ylabel("RSS")

##5.1.5. Simple Linear Regression Code
import statsmodels.api as sm

mod = sm.OLS(y, x)
est = mod.fit()
print(est.summary()) #aquí vemos los detalles del modelo. El coef es > del esperado (debía ser 2)
## al parecer la línea pasa por un b en el eje y, mientras que nuestro ajuste parte de 0
## para arreglar esto se agrega un componente constante
X = sm.add_constant(x)
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary())
##ahora vemos que el coef de x1 está cercano a 2

## 5.1.6. Multiple linear regression
## 5.1.7. Scikit-learn for linear regression

n = 500
beta_0 = 5 
beta_1 = 2
beta_2 = -1
np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ss.norm.rvs(loc=0, scale=1, size=n)

X = np.stack([x_1, x_2], axis = 1)

###PLOT 3D
from mpl_toolkits.mplot3d import Axes3D
fig= plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], y, c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$");

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
lm.fit(X, y)
lm.intercept_ ##devuelve beta_0
lm.coef_[0] ##devuelve beta_1
lm.coef_[1] ##devuelve beta_2

X_0 = np.array([2, 4])
lm.predict(X_0.reshape(1, -1))
lm.score(X, y) ##the model takes imput values X and predict y* values and then compare y* con y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, train_size= 0.5, random_state=1)
lm = LinearRegression(fit_intercept=True)
lm.fit(X_train, y_train)
lm.score(X_test, y_test)





















