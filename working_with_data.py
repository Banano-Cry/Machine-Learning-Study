from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
#dftrain.age.hist(bins=100)
#dftrain['sex'].value_counts().plot(kind='barh')
print(dftrain.value_counts())
#dftrain['class'].value_counts().plot(kind='barh')
plt.show()
print(dftrain.head())
print(dftrain.describe())
