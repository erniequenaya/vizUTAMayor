# years plus interpolation

import os
import datetime
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import pandas
import sqlalchemy
import pymysql

### Contexto
# El siguiente codigo tiene el objetivo de generar CSVs de datos de actualizados-mensualmente para facilitar el proceso de reentramiento de modelos
# De esta forma se ahorra una gran cantidad de líneas de codigo que habria de repetir en cada script de reentramiento para hacer query
# de los datos mas recientes
# Y además se ahorra el costo computacional de hacer query 4 veces (en estricto rigor 3 y 1 vez por modelo) para actualizacion de datos

# Cargando datos locales
# A la fecha de escritura de este codigo, corresponden a registros de datos hasta el 31 de julio
df = pandas.read_csv("../dataPreprocessed.csv",parse_dates=['utc'])
lastTrainDate = df['utc'][-1:]
then = lastTrainDate.to_string(index=False)
# Este script, junto a los de reentramiento de ejecutan a principio de cada mes
# Por lo que los registros en dataPreprocessed siempre deberian estar siempre con un mes de atraso
now = datetime.datetime.now()
now = now.replace(microsecond=0)
print(now)

# Se obtienen datos de la DB para actualizar el csv los datos preprocesados
try:
    credentials = np.genfromtxt("pass",dtype='str')
    engine = sqlalchemy.create_engine("mysql+pymysql://"+credentials[0]+":"+credentials[1]+"@"+credentials[2]+"/"+credentials[3] )
    mydb = engine.connect()
    # La inserción de fechas en la query siempre ha de hacerse como string, los datos tipo datetime pueden user el predeterminado metodo 'str()'
    # Los datos obtenidos de un dataframa han de usar el metodo 'to_string()'
    query = "SELECT * FROM WEATHER_MEASUREMENT WHERE serverDate >= '"+then+"' ;"
    localdf = pandas.read_sql(query,mydb)
except:
    mydb.close() 
    print("error conexion a db")

localdf["utc"] = pandas.to_datetime(localdf["serverDate"],format='%Y-%m-%d %H:%M:%S')
localdf = localdf[['AMBIENT_TEMPERATURE','AIR_PRESSURE','HUMIDITY','utc']]
# Agrupando parametros segun hora para obtener un dataset de 24 * 30o31 filas
localdf = localdf.groupby(pandas.Grouper(key="utc",freq='H')).mean()
### Interpolado
# El rellenado de datos faltantes puede hacerse de una serie de maneras y tienen un alto impacto en la posterior construccion de modelos
# Para mantener la simpleza se ocupa un interpolado lineal tradicional
# hasta que pueda realizarse un web screpping de registros DGAC (csv) y un df.update para incrustar nuestros datos locales sobre estos
localInterpolated = localdf.interpolate(method="linear")

localInterpolated.AMBIENT_TEMPERATURE = localInterpolated.AMBIENT_TEMPERATURE.round(2)
localInterpolated.HUMIDITY = localInterpolated.HUMIDITY.round(2)
localInterpolated.AIR_PRESSURE = localInterpolated.AIR_PRESSURE.round(2)
# Renombrado de columnas, paso importante para no estropear el df.append siguiente, el cual prioriza el nombre de columnas
localInterpolated.columns = ["Ts_Valor","QFE_Valor","HR_Valor"]
localInterpolated = localInterpolated.reset_index()

df =  df.append(localInterpolated, ignore_index=True)
# Guardado de data preprocesada a un csv consumible por cada script de reentramiento
df.to_csv("../dataPreprocessed.csv",index=False)

mydb.close()
engine.dispose()
