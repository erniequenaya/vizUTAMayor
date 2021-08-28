# years plus interpolation

import os
import datetime
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import numpy as np
import pandas
import sqlalchemy
import pymysql

# Cargando datos locales
df = pandas.read_csv("../../data/local/dataPreprocessed.csv")

try:
    credentials = np.genfromtxt("pass",dtype='str')
    engine = sqlalchemy.create_engine("mysql+pymysql://"+credentials[0]+":"+credentials[1]+"@"+credentials[2]+"/"+credentials[3] )
    mydb = engine.connect()
    # Considerando que los datos precargados por csv cuentan con registros hasta el 31 de julio es que hacemos query desde el 1 de agosto
    query = "SELECT * FROM WEATHER_MEASUREMENT WHERE serverDate >= '2021-08-01 00:00:00';"
    # el ultimo dia solo hubo 5442 registros q no alcanzan pa 24 horas
    df = pandas.read_sql(query,mydb)
except:
    mydb.close() 
    print("error conexion a db")
