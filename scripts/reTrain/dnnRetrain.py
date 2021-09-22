#!/root/prediccionService/venv/bin/python

import os
import datetime
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf

def createTimeFeatures (dfToExpand):
    dfToExpand['hour'] = pandas.to_datetime(dfToExpand['utc']).dt.hour
    dfToExpand['day'] = pandas.to_datetime(dfToExpand['utc']).dt.day
    dfToExpand['month'] = pandas.to_datetime(dfToExpand['utc']).dt.month
    dfToExpand['year'] = pandas.to_datetime(dfToExpand['utc']).dt.year
    return dfToExpand

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.legend()
  plt.grid(True)

## Contexto
# El siguiente modelo nace de la necesidad de, valga la redundancia, generar un modelo el cual pueda PRODUCIR PRONOSTICOS climaticos sin necesidad
# de recurrir a, o entender, valores pasados de variables climaticas y las interrelaciones entre éstas respectivamente, pues hasta el momento se 
# desconoce la forma de como alimentar un modelo ML cuyo fin sea realizar predicciones a "largos" periodos en el futuro
# La afirmacion anterior se deriva de la idea de que la definicion de "mañana" u otro momento futuro, consta unicamente de un MOMENTO en el tiempo, 
# es decir, una variable compuesta de hora, dia, mes y año, y por lo tanto, no se cuenta, por ejemplo, con la temperatura de ESE MOMENTO
# Ejemplo: La temperatura del 1 de noviembre de 2021 guarda una estrecha relación con
# la presión QFE del 1 de noviembre de 2021, por lo que sería un perfecto predictor, 
# sin embargo no se cuenta con esta presión, pues el 1 de noviembre de 2021 todavía no ocurre cuando es de interes realizar pronósticos climáticos
# para ese día
# El problema anterior puede apalearse si asumimos que la temperatura del 1 de noviembre de 2021 guarda una estrecha relación en cambio con la
# presión QFE del 1 de noviembre de 2020 (notese el año), o también, con la presión QFE del 31 de octubre de 2021 ("ayer").
# Esta última idea es explorada en el modelo 3 RF y con mayor profundidad en el modelo 4, LSTM

## Carga y creacion de caracteristicas del dataset
df = pandas.read_csv("../dataPreprocessed.csv")
df = createTimeFeatures(df)
df = df[['Ts_Valor','HR_Valor','QFE_Valor','hour','day','month','year']]

## Definiendo variables a predecir y split entre dataset de entrenamiento y test
# Este modelo es "variable-climatica-agnostico", es decir, sus predicciones se construyen unicamente en base a caracteristicas de tiempo
# y no intenta describir ni inferenciar de ninguna forma la relacion entre variables climaticas que la meteorologia tradicional explica
# (como la relacion presion-temperatura)
# Esto con el fin de detectar los comportamientos ciclicos de las variables climaticas y evaluar si estos ciclos son buenos predictores
# para los valores futuros de las variables

varToPred = ['Ts_Valor','HR_Valor','QFE_Valor']
lendf = len(df)
lendf = round(lendf*0.9)
train_df = df[0:lendf]
test_df = df[lendf:len(df)]
train_labels = df[varToPred][0:len(train_df)]
test_labels = df[varToPred][lendf:]

### Normalizado de datos 
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
train_labels = (train_labels - train_mean[varToPred]) / train_std[varToPred]
test_labels = (test_labels - train_mean[varToPred]) / train_std[varToPred]

## Seleccionando variables input
inputVars = ['hour','day','month']
train_df = train_df[inputVars]
test_df = test_df[inputVars]
features = train_df.shape[1]

## Conversión a numpy
train_df = np.array(train_df)
test_df = np.array(test_df)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

### Creando modelo
# El modelo se compone de un stack de 5 capas densas (que quiere decir *densamente conectadas* o *fully connected* en ingles)
# con una cantidad decreciente de neuronas, pues el fin de este modelo es codificar el comportamiento ciclico de las variables para luego
# generar un output pequeño de dimensión 3x1
# El metodo de activación de las capas corresponde a 'relu' abreviación de 'Rectified Linear Unit', el cual permite al modelo
# generalizar de mejor manera los comportamiento de los registros que consume, pues a diferencia de una funcion logistica o sigmoide
# no sufre de problemas de desvanecimiento o explosion de gradiente
# mas info: https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/
nnHM = tf.keras.Sequential()
nnHM.add(tf.keras.layers.Dense(672,input_shape=(features,)))
nnHM.add(tf.keras.layers.Dense(168,activation='relu'))
nnHM.add(tf.keras.layers.Dense(72,activation='relu'))
nnHM.add(tf.keras.layers.Dense(24,activation='relu'))
nnHM.add(tf.keras.layers.Dense(3))

# train_df[['hour','month']].to_numpy()
## Testear que el modelo puede consumir el dataset dada la definicion de dimensionalidad en el input_shape de su primera capa
#nnHM.predict(test_df)

### Compilado
# Se mantienen valores comunes en la practica
# Funcion de perdida: tiene como fin representar y cuantificar el ERROR de una predicción como un valor escalar real, mediante el calculo de la 
# distancia de la prediccion vs el valor del label. Se selecciona el metodo MSE pues es mas punitivo en el modelo considerando el ERROR que otros
# metodos como 'mean_absolute_error', acelerando el entrenamiento
# Optimizador: Motor encargado de propagar cambios en los pesos de las neuronas con el fin de minimizar el ERROR, Adam es un optimizador ampliamente
# utilizado en redes neuonales pues mezcla los optimizadores 'momentum' y 'rmsprop'
# mas info: https://ruder.io/optimizing-gradient-descent/index.html
# Metrics: Metricas para representar la efectividad de prediccion del modelo. Es la principal fuente de información para presenciar la evolución
# del entrenamiento del modelo a traves de los 'epochs' de entremiento junto con la disminución de la función de pérdida
nnHM.compile(
    optimizer=tf.optimizers.Adam(),
    #learning_rate = 0.001
    loss='mse', # 'mean_squared_error',
    metrics=[tf.keras.metrics.MeanSquaredError()] 
)

### Entrenamiento
# Epochs: Cantidad de veces a entrenar el modelo
# Shuffle: Indica si el datastet de entrenamiento debe ser reordenado aleatoriamente en cada epoch, se desactiva esta opción pues tanto las secuencias 
# de los datos de entrada es importante
# Validation_split: Indica la cantidad de datos de entrenamiento que han de reservarse para validar el desempeño de la regresión, pues ete desmpeño
# es un indicador base para definir si el modelo esta mejorando la efectividad de sus predicciones o no
history = nnHM.fit(
    train_df, train_labels,
    epochs=15,
    verbose=1,
    shuffle=False,
    validation_split=0.15
)

nnHM.save('../../../models/dnn')

#plot_loss(history)
#plt.show()

### Testeo
# La caracteristica de ser "variable-climatica-agnostica" le da la ventaja al modelo de generar predicciones para cualquier momento de tiempo
# independiente de que tan en el futuro esté, y a un bajo costo computacional (puesto que la estacionalidad o comportamiento ciclico de las 
# variables climaticas se mantiene inmutable una vez entrenado el modelo)
# Por supuesto, esto trae consigo la desventaja de que el modelo no se puede beneficiar de la informacion de registros meteorologicos nuevos
# pues simplemente no los puede consumir como input
## 
## x = nnHM.predict(test_df)
## 
## stackPreds = pandas.DataFrame(x)
## stackPreds.columns = ['Ts_Valor','HR_Valor','QFE_Valor']
## test_labels = pandas.DataFrame(test_labels)
## test_labels.columns = ['Ts_Valor','HR_Valor','QFE_Valor']
## test_labels = test_labels[varToPred]
## 
## ## Comparar el comportamiento cíclico captado por el modelo vs ciclos reales del set de testeo
## stackPreds.plot(subplots=True)
## test_labels.plot(subplots=True)
## plt.show()
## 
## stackPreds = stackPreds * train_std + train_mean
## test_labels = test_labels * train_std + train_mean
## 
## plt.plot(stackPreds['HR_Valor'])
## plt.plot(test_labels['HR_Valor'])
## plt.legend()
## plt.grid()
## plt.title("Predicciones de humedad relativa - Modelo DNN - Fórmula: HR ~ h + d + m")
## plt.ylabel("Humedad relativa (%)")
## plt.xlabel("Horas en el futuro desde el 31-07 03:00")
## plt.show()
## 
## plt.plot(stackPreds['QFE_Valor'])
## plt.plot(test_labels['QFE_Valor'])
## plt.legend()
## plt.grid()
## plt.title("Predicciones de presión atmosférica - Modelo DNN - Fórmula: QFE ~ h + d + m")
## plt.ylabel("Presión atmosférica (hPac)")
## plt.xlabel("Horas en el futuro desde el 31-07 03:00")
## plt.show()
## 
## plt.plot(stackPreds['Ts_Valor'],label="Predicciones")
## plt.plot(test_labels['Ts_Valor'],label="Test Labels")
## plt.title("Predicciones de temperatura - Modelo DNN - Fórmula: Ts ~ h + d + m")
## plt.ylabel("Temperatura (°C)")
## plt.xlabel("Horas en el futuro desde el 31-07 03:00")
## plt.legend()
## plt.grid()
## plt.show()
## 
## 
## a = ((stackPreds - test_labels) * (stackPreds - test_labels)).mean()
## a

# mean squared error deeper DNN:
# HR_Valor     40.900182
# QFE_Valor     4.978392
# Ts_Valor      1.426796

# HR_Valor     45.708944
# QFE_Valor     3.639670
# Ts_Valor      2.445449
