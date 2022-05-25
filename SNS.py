# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:11:04 2020

@author: Juli Gamboa
"""

# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='white', context='notebook', rc={'figure.figsize':(20,10)})
papas = pd.read_csv("F_SP_ALM_JG.S.csv")
papas.head()
# papas = papas.dropna()
papas.CT.value_counts()
sns.pairplot(papas, hue='CT')
