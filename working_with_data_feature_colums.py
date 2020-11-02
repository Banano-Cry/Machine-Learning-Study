from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

#Creacion del ipunt function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))

        if shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

#Creacion del modelo
def make_model(feature_columns,train_input_fn,eval_input_fn):
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)
    return (linear_est,result)

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #Leemo la data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

#Eliminar un campo del DataFrame
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#Analizar columnas
CATEGORICAL_COLUMS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone'] #Datos categoricos
NUMERIC_COLUMS = ['age','fare'] #datos numericos

feature_columns = []
for feature_name in CATEGORICAL_COLUMS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#Mezclando la data
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#Crear y entrenar el modelo
clear_output()
linear_est,result = make_model(feature_columns,train_input_fn,eval_input_fn)

print(result['accuracy'])

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist',bins=20,title='Predicted probabilities')
plt.show()

resultP = list(linear_est.predict(eval_input_fn))

for i in range(10):
    print("Person:\n",dfeval.loc[i])
    print("Survive:",y_eval.loc[i])
    print("Predic:",resultP[i]['probabilities'][1])
