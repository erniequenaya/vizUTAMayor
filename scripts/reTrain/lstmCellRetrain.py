#!/root/prediccionService/venv/bin/python

import os
import datetime
#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf

def createTimeFeatures (dfToExpand):
    dfToExpand['minute'] = pandas.to_datetime(dfToExpand['utc']).dt.minute
    dfToExpand['hour'] = pandas.to_datetime(dfToExpand['utc']).dt.hour
    dfToExpand['day'] = pandas.to_datetime(dfToExpand['utc']).dt.day
    dfToExpand['month'] = pandas.to_datetime(dfToExpand['utc']).dt.month
    dfToExpand['year'] = pandas.to_datetime(dfToExpand['utc']).dt.year
    return dfToExpand

### Contexto
# El siguiente codigo idealmente ha de ser leído luego de haberse contextualizado con las ideas exploradas en los 3 modelos predecesores

# En pocas palabras, el siguiente codigo es un reintento de generar un modelo el cual pueda producir pronósticos por medio de 
# la detección de ciclos naturales en las variables climáticas (como en el modelo DNN) y además considerando los valores anteriores de 
# estas mismas variables en un periodo de 24 HORAS (como el modelo ARIMA, pero con 24 valores pasados en vez de 48)
# La ventaja de este modelo es que cuenta con mayor información para establecer una sola predicción
# ya que consumira una tabla de 24 filas y 6 columnas para generar 3 pronósticos, uno por variable climatica
# Input [24]x[6] -> Output [1]x[3]
# Para entender de mejor manera los nuevos predictores es que se muestra la fórmula que rige este modelo junto a las fórmulas de modelos anteriores:
# Simplificando para la predicción solo de temperatura, siendo 'n' la representación de la hora actual:
# ARIMA:        Ts(n) ~ Ts(n-1) + Ts(n-2) + ... + Ts(n-48)
# DNN:          Ts(n) ~ h + d + m
# LSTM:         Ts(n) ~ Ts(n-1) + Ts(n-2) + ... + Ts(n-24) + h + d + m

# La definición de predicciones a largos plazo (como e.g. n+24) se construiran de forma autoregresiva, es decir, cada predicción generada por el modelo
# sera utilizada como dato de entrada para generar una nueva prediccion
# De esta forma para este modelo, la matriz Input funcionara como una pila FIFO 
# Cada pronóstico del modelo sera guardado a medida que se descarten los mas antiguos con tal de mantener la pila siempre de tamaño 24x6 
# De este manera, el modelo puede dar 'pasos' indefinidamente en el futuro

## Carga de datos preprocesados y creación de caracteristicas de tiempo
df = pandas.read_csv("../dataPreprocessed.csv")
df = createTimeFeatures(df)
dates_df = df.pop('minute')
dates_df = df.pop('year')
dates_df = df.pop('utc')

## Normalizado
df2 = df.copy()
train_mean = df.mean()
train_std = df.std()
df = (df - train_mean) / train_std

## Creación datasets de entrenamiento y testeo
lendf = len(df)
lendf = round(lendf*0.9)
train_df = df[0:lendf]
train_labels = df[['Ts_Valor','HR_Valor','QFE_Valor']][0:lendf]
test_df = df[lendf:len(df)]
test_labels = df[['Ts_Valor','HR_Valor','QFE_Valor']][lendf:len(df)]

## Transformando a numpy 
train_df = train_df.to_numpy()
test_df = test_df.to_numpy()
train_labels = train_labels.to_numpy()
test_labels = test_labels.to_numpy()

## Implementando sistema de ventanas

## mas info: https://www.gitmemory.com/issue/tensorflow/tensorflow/44592/752315411
##           https://www.tensorflow.org/tutorials/structured_data/images/lstm_1_window.png
winSize = 24 
numFeatures = 6 
train_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(train_df,train_labels,length=winSize)
test_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(test_df,test_labels,length=winSize)

## Comprobando funcionamiento de sistema de ventanas
# np.set_printoptions(suppress=True)
# X,y = train_gen[0]
# print(f'Given the Array: \n{X[0]}')
# print(f'Predict this y: \n {y[0]}')

## Creación de arquitectura del modelo
# El modelo consiste en una capa de células LSTM, una capa Flatten para disminuir la dimensionalidad de las predicciones
# y una capa Dense para convertir los outputs de las capas anteriores en el output deseado final (3 variables) 
# El uso de células LSTM por separado 
# (puesto que la implementacion de keras.layers.LSTM en estricto rigor es un estructura LSTM+RNN que reduce dimensionalidad)
# Permite que las celulas tengan 'memorias' independientes que aportan valor de formas iguales  
# Añadir mas capas al modelo tiende a acelarar el overfitting y disminuir la varianza de las predicciones 
# 
nnHM = tf.keras.Sequential()
# nnHM.add(tf.keras.layers.LSTMCell(432,input_shape=(winSize,numFeatures,),return_state=True))      # 288 due to 6*24*2days
nnHM.add(tf.keras.layers.LSTMCell(365,input_shape=(winSize,numFeatures,),stateful=True))      # 288 due to 6*24*2days
# nnHM.add(tf.keras.layers.Dense(72))
nnHM.add(tf.keras.layers.Flatten())
# # nnHM.add(tf.keras.layers.Dense([24,3]))   # this output format doesnt work as the TSgen was not defined with with large-windows outputs
nnHM.add(tf.keras.layers.Dense(3))

nnHM.compile(
    optimizer=tf.optimizers.Adam(),
    #loss='mae', # 'mean_absolute_error',
    loss='mse', # o 'mean_squared_error', permite un entrenamiento mas rapido del modelo
    metrics=[tf.keras.metrics.MeanSquaredError()]
)
history = nnHM.fit_generator(
    train_gen,
    epochs=7,      # basadose netamente en experiencia (o sin apoyo teorico), mantener este valor entre 7 y 15, a mas de 15 solo se hace overfit
    verbose=1,
    shuffle=False,
    validation_data=test_gen
)

## Salvando el modelo
nnHM.save("../../../models/lstm")
# nnHM = tf.keras.models.load_model("../models/lstm")

## Elaboracion de una prediccion 
# el siguiente codigo se muestra solo con proposito de debugging, por lo tanto no se ha de descomentar cuando este script sea
# implementado en el servidor

# Se obtiene los ultimos 24 registros del set de entrenamiento, se formatean como 1 ventana y se predice en base a esta ventana
# Este bloque de codigo contiene las bases de la implementacion de prediccion multipaso o autoregresiva en el script *../lstmPredictions.py*
lastTrainBatch = train_df[-24:]
lastTrainBatch = np.array(lastTrainBatch)
# now reshape this lastTrainBatch to meet what input_shape the model expects
lastTrainBatch = lastTrainBatch.reshape((1,winSize,numFeatures))
lstmPred = nnHM.predict(lastTrainBatch,verbose=1)
lstmPred

# Notese la diferencia contra un test de prediccion estandar
x = nnHM.predict_generator(test_gen,verbose=1)
x.shape
# Las ventanas pasadas no son multipaso pues el test_gen alimenta la prediccion con ventanas independiente de las predicciones generadas por el modelo
# por lo tanto, el modelo no "camina" por sobre sus propias predicciones

# Para poder mostrar las predicciones en web, estas deben ser de-normalizadas (regresadas a su rango de valores originales) y convertidas a un dataframe
# para facilitar su stacking
# De-normalizacion
X = pandas.DataFrame(x)
X.columns = ['Ts_Valor','HR_Valor','QFE_Valor']
varToPred = ['Ts_Valor','HR_Valor','QFE_Valor']
X = X*train_std[varToPred]+train_mean[varToPred]
X

# edefine test_df to fit preds
lendf = lendf+24
test_df = df[lendf:len(df)]
test_labels = df[lendf:len(df)]
test_labels = test_labels*train_std+train_mean
test_labels = test_labels.reset_index(drop=True)

plt.plot(test_labels[['Ts_Valor']][24:],label='testDF')
plt.plot(X[['Ts_Valor']],label='dense+lstm')
plt.grid()
plt.legend()
plt.show()

a = (X - test_labels).mean()
a


# retake df dates for 'now' value when testdf sizes are altered
test_df = df2[lendf:len(df)]

# Recuerda que 1 prediccion requiere 24 horas de datos, por lo tanto, el 'now' para el modelo corresponde a la hora siguiente de ese tramo de
# 24 horas de datos
# when 0.995 split
# now = pandas.to_datetime('2021-07-26 11:00:00')
# when 0.9 split
now = pandas.to_datetime('2021-03-23 06:00:00')

df3 = train_df[-24:]
stackPreds = pandas.DataFrame()
lastTrainBatch = train_df[-24:]
lastTrainBatch = np.array(lastTrainBatch)
# now reshape this lastTrainBatch to meet what input_shape the model expects
lastTrainBatch = lastTrainBatch.reshape((1,winSize,numFeatures))
x = nnHM.predict(lastTrainBatch)

def norm(value,index):
    #value = (value - modelMean[index]) / modelStd[index]
    value = (value - train_mean[index]) / train_std[index]
    return value

for i in range(0,72):
    delta = now + datetime.timedelta(0,i*3600)
    #temp = np.array([x,y,z,delta.hour,delta.day,delta.month],dtype="float32")
    # temp = np.array([x,y,z,norm(delta.hour,3),norm(delta.day,4),norm(delta.month,5)],dtype="float32")
    temp = np.array([x[0,0],x[0,1],x[0,2],norm(delta.hour,3),norm(delta.day,4),norm(delta.month,5)],dtype="float32")
    stackPreds = stackPreds.append(pandas.DataFrame(temp).transpose())
    # df3 = np.vstack((df3[0],temp))
    # df4 = np.vstack((df4,temp))
    cde = np.reshape(df3,(24,6))
    df3 = np.vstack((cde,temp))
    df3 = np.delete(df3, (0), axis=0)
    df3 = np.reshape(df3,(-1,24,6))
    x = nnHM.predict(df3)

stackPreds = stackPreds.reset_index(drop=True)
stackPreds.columns = ['Ts_Valor','HR_Valor','QFE_Valor','hour','day','month']
stackPreds = stackPreds*train_std+train_mean
stackPreds

predTest = nnHM.predict(test_gen,verbose=1)
varToPred = ['Ts_Valor','HR_Valor','QFE_Valor']
predTest = predTest*train_std[varToPred].to_numpy()+train_mean[varToPred].to_numpy()

df = pandas.read_csv("../dataPreprocessed.csv")
df = createTimeFeatures(df)
dates_df = df.pop('utc')
dates_df = df.pop('minute')
dates_df = df.pop('year')
test_df = df[lendf:len(df)]
test_labels = df[['Ts_Valor','HR_Valor','QFE_Valor']][lendf:len(df)]
test_labels = test_labels.reset_index(drop=True)

plt.plot(test_labels['Ts_Valor'], label='testLabels')
plt.plot(predTest[0:,0],label='predTestDF')
plt.plot(stackPreds['Ts_Valor'],label='predAutonoma')
plt.legend()
plt.grid()
plt.show()

plt.plot(test_labels['HR_Valor'], label='testLabels')
plt.plot(predTest[0:,1],label='predTestDF')
plt.plot(stackPreds['HR_Valor'],label='predAutonoma')
plt.legend()
plt.grid()
plt.show()

plt.plot(test_labels['QFE_Valor'], label='testLabels')
plt.plot(predTest[0:,2],label='predTestDF')
plt.plot(stackPreds['QFE_Valor'],label='predAutonoma')
plt.legend()
plt.grid()
plt.show()

a = stackPreds - test_labels[:72]
a.mean()


