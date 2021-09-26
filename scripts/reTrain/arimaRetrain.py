# este codigo es solo de muestra, puesto que no existe un modelo ARIMA entrenado el cual sea llamado cada 24 horas para realizar predicciones de forma regular
# en cambio, cada 24 horas se construyen modelos ARIMA nuevos para realizar las predicciones

import os
import datetime
#import matplotlib.pyplot as plt
import numpy as np
import pandas
import sqlalchemy
import pymysql
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
#import joblib

df = pandas.read_csv("../dataPreprocessed.csv",parse_dates=['utc'],index_col=['utc'])
# Manteniendo últimos dos dias de registros
lendf = len(df)
df = df[-148:-72]
df = df.interpolate(method="linear")
series = df.copy()
series.index = series.index.to_period('H')

future = 72

stackPreds = pandas.DataFrame()
model = ARIMA(series['Ts_Valor'], order=(24,1,0))
model_fit = model.fit()
stackPreds['Ts_Valor'] = model_fit.forecast(future)
model = ARIMA(series['HR_Valor'], order=(24,1,0))
model_fit = model.fit()
stackPreds['HR_Valor'] = model_fit.forecast(future)
model = ARIMA(series['QFE_Valor'], order=(24,1,0))
model_fit = model.fit()
stackPreds['QFE_Valor'] = model_fit.forecast(future)

df = pandas.read_csv("../dataPreprocessed.csv",parse_dates=['utc'],index_col=['utc'])
test_labels = df[-72:]
a = mean_absolute_error(stackPreds[['Ts_Valor','HR_Valor','QFE_Valor']],test_labels[['Ts_Valor','HR_Valor','QFE_Valor']],multioutput='raw_values')
a
