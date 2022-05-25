# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:45:22 2020

@author: Diez
"""

import numpy as np
from scipy.integrate import odeint
#matplotlib inline
import matplotlib.pyplot as plt

def firstorder (y, t, K, u):
    tau = 5.0
    dydt = (-y + K*u)/tau
    return dydt

t = np.linspace(0,10,11)
K = 2.0
u = np.zeros(len(t))
u[3:] = 1.0 #different u values
y0 = 0
print(u)

ys = np.zeros (len(t))
ys[0] = y0
for i in range (len(t)-1):
    ts = [t[i], t[i+1]]
    y = odeint (firstorder, y0, ts, args=(K,u[i]))
    y0 = y[1]
    ys[i+1] = y0
    print (y[1])
plt.plot(t, ys)