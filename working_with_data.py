from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #Leemo la data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

#Eliminar un campo del DataFrame
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#Visualizar las primeras 5 fila del DataFrame
print(dftrain.head())

#Visualizar unas caracteristicas calculadas, como la media de los datos o su varianza
print(dftrain.describe())

#Visualizar el tamaño del DataFrame
print(dftrain.shape)

#Visualizar un histograma con los datos de la edad
dftrain.age.hist(bins=100)
plt.show()

#Visualizar en una grafica de barra los datos del sexo, por no ser numerico se usa value_counts()
dftrain['sex'].value_counts().plot(kind='barh')
#dftrain['class'].value_counts().plot(kind='barh')
plt.show()

#Concatenamos los DataFrame para agruparlos por sexo, porteriormente obtener la media y graficarlo
pd.concat([dftrain, y_train], axis = 1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()
