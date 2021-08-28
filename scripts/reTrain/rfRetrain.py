#!/root/prediccionService/venv/bin/python
# as we cant export R models due to massacring one cpu core lets export all as python models

import os
import datetime
#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
import joblib

def createTimeFeatures (dfToExpand):
    dfToExpand['hour'] = pandas.to_datetime(dfToExpand['utc']).dt.hour
    dfToExpand['day'] = pandas.to_datetime(dfToExpand['utc']).dt.day
    dfToExpand['month'] = pandas.to_datetime(dfToExpand['utc']).dt.month
    dfToExpand['year'] = pandas.to_datetime(dfToExpand['utc']).dt.year
    return dfToExpand

## Contexto
# El siguiente codigo es un reintento de generar un modelo el cual pueda producir pronósticos por medio de la detección de ciclos naturales en las
# variables climáticas (como en el modelo DNN) y además considerando los valores INMEDIATAMENTE ANTERIORES de estas mismas variables
# (como una versión limitada del modelo ARIMA, pues se toma una hora de valores pasados en vez de 48)
# Para entender de mejor manera los nuevos predictores es que se muestra la fórmula que rige este modelo junto a las fórmulas de modelos anteriores:
# Siendo 'n' la representación de la hora actual
# ARIMA:        Ts(n) = Ts(n-1) + Ts(n-2) + ... + Ts(n-48)
# DNN:          Ts(n) = h + d + m                               # notese la ausencia de algún comoponente hora-dependiente
# RF(nuevo):    Ts(n) = Ts(n-1) + h + d + m
# Hipoteticamente se espera que esta relación 'label~predictor' sea capaz de detectar el comportamiento cíclico de las variables climáticas
# (mediante h+d+m) y capaz también de influenciar cada prediccion mediante la temperatura que le precede (mediante Ts(n-1))
# La definición de predicciones a largos plazo (como e.g. n+24) se construiran de forma autorregresiva, es decir, cada predicción generada por el modelo
# sera utilizada como dato de entrada para generar una nueva prediccion
# Esta forma de evaluación de modelos se ha convertir en la forma 'de jure' de testeo de modelos, pues engloba perfectamente los requerimientos
# del proyecto (pronósticos de 24 a 168 horas)
# De esta forma es que para este modelo se construyen nuevas columnas de datos de entrenamiento correspondiente a registros de e.g. Temperatura
# de hace una hora ( Ts(n-1) ), a esta nueva columna se le llamara 'offset'

# Random Forest no calcula ningun coeficiente de regresion mas que el promedio de sus arboles para determinar una prediccion
# por lo tanto no es sensible a diferencias de varianza o promedio en los datos, no requiriendo normalizado de estos
# Esta caracteristica lo vuelve un algoritmo ideal para la produccion de lineas-base de predicción, pues es capaz de obtener modelos
# funcionales de forma expedita en tiempo de programación y recursos computacionales

# Carga y creacion de caracteristicas del dataset
df = pandas.read_csv("../dataPreprocessed.csv")
df = createTimeFeatures(df)
dates_df = df.pop('utc')

## Creando offset
topred = df[['Ts_Valor','HR_Valor','QFE_Valor']]
# Se ignora el primer registro del dataset pues no existen predictores para las 00:00 del 1 de enero de 2018
topred = topred[1:]
# Y el ultimo registro del dataset original pues no existe un label para las 00:00 del 1 de agosto de 2021
df = df[:-1]

## Dividiendo dataset
lendf = len(df)
lendf = round(lendf*0.9)
train_df = df[0:lendf]
train_labels = topred[0:lendf]
test_df = df[lendf:len(df)]
test_labels = topred[lendf:len(df)]

## Notese la cantidad de columnas del dataset de entrenamiento, este dataset es el más grande hasta ahora usado
train_df.shape
train_labels.shape

## Creación y entrenamiento del modelo
rf = RandomForestRegressor()
rf.fit(train_df,train_labels)

# Salvando modelo
joblib.dump(rf,'../../../models/rf')

### El siguiente código se adjunta con fines unicamente de testeo, por lo tanto no debe ser ejecutado mediante scripts ###
# Y además para explicar el nuevo mecanismo de generación de predicciones a largo plazo

### Evaluamos
## Procedemos a generar predicciones retroalimentadas
## 'now' corresponde a la fecha desde la cual se deben empezar a construir las predicciones, esta vez
## es definido según la última fecha vista por el dataset de entrenamiento ('x'), aunque puede tomar cualquier fecha futura
# now = pandas.to_datetime('2021-03-23 05:00:00')
# x = rf.predict(train_df[-1:])
# stackPreds = pandas.DataFrame()

### Para predecir mas horas basta con cambiar 72 a un valor mas alto
# for i in range(0,72):
#     # Mediante cada iteración, se agregan 3600 segundos (1 hora) a la fecha definida por 'now'
#     delta = now + datetime.timedelta(0,i*3600)
#     # Se guarda en una variable temporal, las predicciones generadas por el modelo y junto a la nueva fecha 'delta'
#     temp = np.array([x[0,0],x[0,1],x[0,2],delta.hour,delta.day,delta.month,delta.year],dtype="float32")
#     # Se añade al stack de predicciones la prediccion generada para esta iteración
#     stackPreds = stackPreds.append(pandas.DataFrame(temp).transpose())
#     # Formateando el registro para alimentar el modelo 
#     temp_RE = np.reshape(temp,(-1,7))
#     x = rf.predict(temp_RE)
# 
# stackPreds = stackPreds.reset_index(drop=True)
# test_df = test_df.reset_index(drop=True)
#  
# plt.plot(test_df['Ts_Valor'],label='testDF')
# plt.plot(stackPreds[0],label='stack')
# plt.legend()
# plt.grid()
# plt.show()
# 
# plt.plot(test_df['HR_Valor'],label='testDF')
# plt.plot(stackPreds[1],label='stack')
# plt.legend()
# plt.grid()
# plt.show()
# 
# plt.plot(test_df['QFE_Valor'],label='testDF')
# plt.plot(stackPreds[2],label='stack')
# plt.legend()
# plt.grid()
# plt.show()
#  
# stackPreds.columns = ['Ts_Valor','HR_Valor','QFE_Valor','hour','day','month','year']
# a = ((stackPreds - test_df[:72]))
# a.mean()
# 
# Ts_Valor     1.027403
# HR_Valor     3.147222
# QFE_Valor   -0.225665