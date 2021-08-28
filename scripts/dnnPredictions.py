#!/root/prediccionService/venv/bin/python

import os
import datetime
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf
import sqlalchemy
import pymysql

def createTimeFeatures (dfToExpand):
    # A pesar de estar definida como una funcion que crea y retorna una copia del dataset que se le es proveida
    # aun así el dataset original es modificado, no se por que
    dfToExpand['hour'] = pandas.to_datetime(dfToExpand['utc']).dt.hour
    dfToExpand['day'] = pandas.to_datetime(dfToExpand['utc']).dt.day
    dfToExpand['month'] = pandas.to_datetime(dfToExpand['utc']).dt.month
    dfToExpand['year'] = pandas.to_datetime(dfToExpand['utc']).dt.year
    return dfToExpand

# El modelo DNN existente solo necesita horas, días y minutos arbitrarios para poder generar un pronóstico relativamente fiable 
# -MSE de 41, 5 y 1.5 para HR, QFE y TS respectivamente-
# por lo tanto si se desea hacer pronostico del clima para las proximas 24 horas, basta con obtener la fecha y hora actual, generar los pasos de tiempo
# de una hora y descomponerla en caracteristicas de tiempo significativas
# aunque dado el input con el que fue entrenado el modelo, se le puede ser ingresado cualquier momento en el tiempo arbitrario sin necesidad de ser
# consecutivo

# hour day month in that order
# Generando fechas futuras
now = datetime.datetime.now()
stackPredictors = pandas.DataFrame()
for i in range(72):
    delta = now + datetime.timedelta(0,i*3600)
    temp = np.array([delta.hour,delta.day,delta.month],dtype="float32")
    stackPredictors = stackPredictors.append(pandas.DataFrame(temp).transpose())
    
stackPredictors = stackPredictors.reset_index(drop=True)

# Normalizando manualmente
# el modelo fue entrenado con valores normalizados generados en base a metricas -promedio, desviacion estandar- del dataset de 
# entrenamiento, por lo tanto, estas metricas se traspasan a los predictandos(?) actuales para respetar los valores
# con los que fue entrenado el modelo

modelMean = [18.833937942048827, 67.28831089816715, 1009.0120921743098, 11.500418282759146, 15.734086242299794, 6.521370446421781]
modelStd = [3.0178459228432164, 7.216205483706377, 2.2937924094103446, 6.92221927192334, 8.801762436194657, 3.448822871574395]

stackPredictors = (stackPredictors - modelMean[3:]) / modelStd[3:]

# Cargado de modelo
# dnnMultivar = tf.keras.models.load_model('../../pyCNN/deepNN-multiV2')
dnnMultivar = tf.keras.models.load_model('../../models/dnn/')
# Realizando pronosticos
stackPreds = dnnMultivar.predict(stackPredictors)
stackPreds = pandas.DataFrame(stackPreds)
# Construyendo tabla de predicciones final, conteniendo predictores y Predicciones
stackPreds = pandas.concat([stackPreds,stackPredictors],axis=1)
# De-normalizando valores
stackPreds = stackPreds * modelStd + modelMean
# El año no se considero en el entrenamiento pues disminuia la varianza de las predicciones demasiado
stackPreds['year'] = now.year
stackPreds.columns = ['AMBIENT_TEMPERATURE','HUMIDITY','AIR_PRESSURE','hour','day','month','year']

#stackPreds.plot(subplots=True)
#plt.plot(df4[0,0:24,1])
#plt.plot(df4[0,0:24,2])
#plt.grid()
#plt.show()

stackPreds.AMBIENT_TEMPERATURE = stackPreds.AMBIENT_TEMPERATURE.round(3)
stackPreds.HUMIDITY = stackPreds.HUMIDITY.round(3)
stackPreds.AIR_PRESSURE = stackPreds.AIR_PRESSURE.round(3)
stackPreds.hour = stackPreds.hour.astype(int)
stackPreds.day = stackPreds.day.astype(int)
stackPreds.month = stackPreds.month.astype(int)

stackPreds['utc'] =stackPreds.year.astype(str)+'-'+stackPreds.month.astype(str)+'-'+stackPreds.day.astype(str)+' '+stackPreds.hour.astype(str)+':00:00'
stackPreds['utc'] = pandas.to_datetime(stackPreds['utc'])

try:
    #credentials = np.genfromtxt("../viz/scripts/pass",dtype='str')
    credentials = np.genfromtxt("pass",dtype='str')
    engine = sqlalchemy.create_engine("mysql+pymysql://"+credentials[0]+":"+credentials[1]+"@"+credentials[2]+"/"+credentials[3] )
    mydb = engine.connect()
    stackPreds.to_sql('dnnPredictions',mydb,if_exists='append',index=False)
    query = "ALTER TABLE dnnPredictions ADD id INT PRIMARY KEY AUTO_INCREMENT;"
    mydb.execute(query)
except:
    mydb.close() #close the connectionexcept Exception as e:
    print('Error en conexion a base de datos')

mydb.close()
engine.dispose()
## script ready to be callbable
