#!/root/prediccionService/venv/bin/python
# years plus interpolation

import os
import datetime
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import pandas
import sqlalchemy
import pymysql

def createTimeFeatures (dfToExpand):
    # despite explicitly returning a dataframe as a parameter, this function still modifies the original dataframe, idk why
    # La causa es debido al tratamiento ViewVScopy de pandas
    dfToExpand['hour'] = pandas.to_datetime(dfToExpand['utc']).dt.hour
    dfToExpand['day'] = pandas.to_datetime(dfToExpand['utc']).dt.day
    dfToExpand['month'] = pandas.to_datetime(dfToExpand['utc']).dt.month
    dfToExpand['year'] = pandas.to_datetime(dfToExpand['utc']).dt.year
    return dfToExpand

# Cargando datos preprocesados
df = pandas.read_csv("../../data/local/dataPreprocessed.csv")

# Cargando datos de sensores
try:
    credentials = np.genfromtxt("pass",dtype='str')
    engine = sqlalchemy.create_engine("mysql+pymysql://"+credentials[0]+":"+credentials[1]+"@"+credentials[2]+"/"+credentials[3] )
    mydb = engine.connect()
    # Considerando que los datos precargados por csv cuentan con registros hasta el 31 de julio es que hacemos query desde el 1 de agosto
    query = "SELECT * FROM WEATHER_MEASUREMENT WHERE serverDate >= '2021-08-01 00:00:00';"
    # el ultimo dia solo hubo 5442 registros q no alcanzan pa 24 horas
    localDf = pandas.read_sql(query,mydb)
except:
    mydb.close() 
    print("error conexion a db")


#localDf = pandas.read_sql(query,mydb)
#localDf = localDf.groupby(pandas.Grouper(key="serverDate",freq='H')).mean()

localDf["utc"] = pandas.to_datetime(localDf["serverDate"],format='%Y-%m-%d %H:%M:%S')
#df = createTimeFeatures(df)
localDf = createTimeFeatures(localDf)
localDf = localDf.groupby(['year','month','day','hour']).mean()
#localDf.plot(subplots=True) 
#plt.show()

#df.set_index(['year','month','day','hour'],inplace=True)
train_dates = df.pop('utc')
# the shorter the better
localDf.columns = ['Ts_Valor','QFE_Valor','HR_Valor']
localDf = localDf.reset_index()

#cols=["year","month","day"]
#df.loc[:,'Date'] = pd.to_datetime(df.Date.astype(str)+' '+df.Time.astype(str))
# Como el agrupamiento de los datos por hora destruyo los registro tipo 'date' del dataset original hay que rearmmarlos
localDf['utc'] =localDf.year.astype(str)+'-'+localDf.month.astype(str)+'-'+localDf.day.astype(str)+' '+localDf.hour.astype(str)+':00:00'
localDf['utc'] = pandas.to_datetime(localDf['utc'])
#localDf.plot(subplots=True)
#plt.show()
# A pesar de que han pasado 4 días desde 31 de julio, y por lo tanto 96 horas, solo contamos con 76 registros, esto se debe al downtime de los sensores
# por lo tanto, primero debemos considerar la variable 'utc' como una series de tiempo, inferir la ausencia de registros, el periodo de los registros
# y luego rellenar mediante interpolado
# la siguiente funcion es tan poderosa que se puede aplicarse descaradamente sobre el dataset inmediatamente despues de la query
localDf = localDf.groupby(pandas.Grouper(key="utc",freq='H')).mean()    
# por lo tanto, hacemos interpolado
localDf = localDf.interpolate(method = 'linear')

# Reiniciamos index para poder concatenarlo con el DF pre-preprocesado
localDf = localDf.reset_index()
# Seleccionames y ordenamos finalmente, las columnas a mantener para que calzen con el formato de dataset pre-preprocesado 'df'
cols=['Ts_Valor','HR_Valor','QFE_Valor','utc']
localDf = localDf[cols]

localDf['Ts_Valor'] = localDf.Ts_Valor.round(2)
localDf['HR_Valor'] = localDf.HR_Valor.round(2)
localDf['QFE_Valor'] = localDf.QFE_Valor.round(2)

df = df.append(localDf)
df = x.reset_index(drop=True)

