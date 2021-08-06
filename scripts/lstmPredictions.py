#!/root/prediccionService/venv/bin/python
# Predictor maker
# To be run at least one hour after retraining
import os
import datetime
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf
import sqlalchemy
import pymysql
#import mysql.connector as connection
#from sklearn.preprocessing import MinMaxScaler

try:
    # para rescatar las ultimas 24 horas: 24 * 60 * 12
    # credentials = np.genfromtxt("../viz/scripts/pass",dtype='str')
    credentials = np.genfromtxt("pass",dtype='str')
    engine = sqlalchemy.create_engine("mysql+pymysql://"+credentials[0]+":"+credentials[1]+"@"+credentials[2]+"/"+credentials[3] )
    mydb = engine.connect()
    #stackPreds = pandas.read_sql(query,mydb)
    query = "SELECT * FROM WEATHER_MEASUREMENT ORDER BY ID DESC LIMIT 17280;"
    # el ultimo dia solo hubo 5442 registros q no alcanzan pa 24 horas
    df = pandas.read_sql(query,mydb)
except:
    mydb.close() 
    print("error conexion a db")

df = df.iloc[::-1] # dando vuelta el dataframe
df["utc"] = pandas.to_datetime(df["serverDate"],format='%Y-%m-%d %H:%M:%S')
df = df[['AMBIENT_TEMPERATURE','AIR_PRESSURE','HUMIDITY','utc']]
# generando variables de tiempo para alimentar los modelos
# df['minutes'] = pandas.to_datetime(df['utc']).dt.minute
df['hour'] = pandas.to_datetime(df['utc']).dt.hour
df['day'] = pandas.to_datetime(df['utc']).dt.day
df['month'] = pandas.to_datetime(df['utc']).dt.month
# agrupando parametros segun hora para obtener un dataset de 24 filas
df2 = df.groupby('hour').mean()
df2['hour'] = df2.index
row = df2.mean()

# basta con que se genere 1 registro cada una hora para poder alimentar el modelo
##### solo por ahora, rellenamos los periodos de las ultimas 24 horas faltantes con promedios de los otros registros ######
df3 = pandas.DataFrame(columns=row.keys())
for i in range(24):
    if i in df2.hour:
        print(i)
    else:
        row.hour = i
        df3 = df3.append(row,ignore_index=True)

df2 = df2.append(df3)
df2 = df2.sort_values('hour')
df2 = df2.reset_index(drop=True)
#df2.plot(subplots=True)
#plt.show()
##### solo por ahora, rellenamos los periodos de las ultimas 24 horas faltantes con promedios de los otros registros ######

# Normalizando manualmente
# el modelo fue entrenado con valores normalizados generados en base a metricas -promedio, desviacion estandar- del dataset de 
# entrenamiento, por lo tanto, estas metricas se traspasan a los predictandos(?) actuales para respetar los valores
# con los que fue entrenado el modelo

#modelMean = {'AMBIENT_TEMPERATURE': 18.833937942048827, 'HUMIDITY': 67.28831089816715, 'AIR_PRESSURE': 1009.0120921743098, 'hour': 11.500418282759146, 'day': 15.734086242299794, 'month': 6.521370446421781}
#modelStd = {'AMBIENT_TEMPERATURE': 3.0178459228432164, 'HUMIDITY': 7.216205483706377, 'AIR_PRESSURE': 2.2937924094103446, 'hour': 6.92221927192334, 'day': 8.801762436194657, 'month': 3.448822871574395}
modelMean = [18.833937942048827, 67.28831089816715, 1009.0120921743098, 11.500418282759146, 15.734086242299794, 6.521370446421781]
modelStd = [3.0178459228432164, 7.216205483706377, 2.2937924094103446, 6.92221927192334, 8.801762436194657, 3.448822871574395]
# you can straight substract a list from dataframe without giving key values!
df3 = df2[['AMBIENT_TEMPERATURE','HUMIDITY','AIR_PRESSURE','hour','day','month']]
df3 = (df3 - modelMean) / modelStd
df3 = df3.to_numpy()
# reshape a valores actuales
df3 = np.reshape(df3,(-1,24,6))
# df3 just became a 3 dimensional array jesus...

# Cargado de modelos, 1 por variable
tsModel = tf.keras.models.load_model('../dense+lstmMnormTs/')
qfeModel = tf.keras.models.load_model('../dense+lstmMnormQFE/')
#hrModel = tf.keras.models.load_model('./dense+lstmHRv2/') # to load with a proper model

#x = tsModel.predict(df3) # array([[0.0564339]], dtype=float32), meanAsDict -0.5708247, meanAsList 1.0230861
#y = qfeModel.predict(df3) # array([[2.6169283]], dtype=float32), meanAsDict 2.0415623, meanAsList -2.3020828
#z = 1.5
#########
#x = x*modelStd[0]+modelMean[0]

# Implementando funcion normalizadora
def norm(value,index):
    value = (value - modelMean[index]) / modelStd[index]
    return value

# implementando autoregresion (windowing system manual)

#norm(now.hour,3)
#df4 = np.array([x,y,z,now.hour,now.day,now.month],dtype="float32")
#df4 = np.array([x,y,z,norm(now.hour,3),norm(now.day,4),norm(now.month,5)],dtype="float32")
now = datetime.datetime.now()
stackPreds = pandas.DataFrame()
x = tsModel.predict(df3)
y = 1.5
z = qfeModel.predict(df3)
#stackPreds = pandas.DataFrame(columns=row.keys())
# para predecir mas horas basta con cambiar 12 a 168
for i in range(0,72):
    delta = now + datetime.timedelta(0,i*3600)
    #temp = np.array([x,y,z,delta.hour,delta.day,delta.month],dtype="float32")
    temp = np.array([x,y,z,norm(delta.hour,3),norm(delta.day,4),norm(delta.month,5)],dtype="float32")
    stackPreds = stackPreds.append(pandas.DataFrame(temp).transpose())
    # df3 = np.vstack((df3[0],temp))
    # df4 = np.vstack((df4,temp))
    cde = np.reshape(df3,(24,6))
    df3 = np.vstack((cde,temp))
    df3 = np.delete(df3, (0), axis=0)
    df3 = np.reshape(df3,(-1,24,6))
    x = tsModel.predict(df3)
    y = 1.5
    z = qfeModel.predict(df3)

#modelStd[0]+modelMean[0]

#plt.plot(df4[0,0:24,0])
#plt.plot(df4[0,0:24,1])
#plt.plot(df4[0,0:24,2])
#plt.grid()
#plt.show()

stackPreds.columns = ['AMBIENT_TEMPERATURE','HUMIDITY','AIR_PRESSURE','hour','day','month']
stackPreds = stackPreds.reset_index()
stackPreds = stackPreds[['AMBIENT_TEMPERATURE','HUMIDITY','AIR_PRESSURE','hour','day','month']]

#stackPreds = stackPreds.reset_index()

# de-normalizando predicciones
stackPreds=stackPreds*modelStd+modelMean
# stackPreds.plot(subplots=True)
# plt.show()
# plt.plot(stackPreds['AMBIENT_TEMPERATURE'])

stackPreds.AMBIENT_TEMPERATURE = stackPreds.AMBIENT_TEMPERATURE.round(3)
stackPreds.HUMIDITY = stackPreds.HUMIDITY.round(3)
stackPreds.AIR_PRESSURE = stackPreds.AIR_PRESSURE.round(3)
stackPreds.hour = stackPreds.hour.astype(int)
stackPreds.day = stackPreds.day.astype(int)
stackPreds.month = stackPreds.month.astype(int)
stackPreds['year'] = now.year
stackPreds.year = stackPreds.year.astype(int)

stackPreds['utc'] =stackPreds.year.astype(str)+'-'+stackPreds.month.astype(str)+'-'+stackPreds.day.astype(str)+' '+stackPreds.hour.astype(str)+':00:00'
stackPreds['utc'] = pandas.to_datetime(stackPreds['utc'])

try: 
    stackPreds.to_sql('predictions',mydb,if_exists='replace',index=False)
    query = "ALTER TABLE predictions ADD id INT PRIMARY KEY AUTO_INCREMENT;"
    mydb.execute(query)
except:
    print("error en conexion a db")

#print(stackPreds)
mydb.close()
engine.dispose()

