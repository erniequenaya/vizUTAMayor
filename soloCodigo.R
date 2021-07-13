library(RMySQL)
library(lubridate)
library(hms)
library(ggplot2)
library(dplyr)

# Configurando conexion a DB
# proveer contraseña mediante cuadro de dialogo
mydb = dbConnect(MySQL(), user='admin_mysql', password=.rs.askForPassword("Contraseña DB:"), dbname='weather', host='192.168.50.176')
weather = dbSendQuery(mydb, "select * from WEATHER_MEASUREMENT")
df = fetch(weather, n=-1)

# importar como csv
# en caso de no contar con conexion a DB o de querer repetir los graficos expuestos el 25 de junio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
df <- read.csv("weather.csv",header = TRUE, sep = ",")

# transformacion de mormento de creacion de filas a un formato operable por R
# mas info: https://raw.githubusercontent.com/rstudio/cheatsheets/master/lubridate.pdf
df2<-df

# en caso de haber cargado dataset desde db correr
df2$dateUTC<-ymd_hms(df$serverDate)

# en caso de haber cargado dataset mediante csv adjunto
# df2$dateUTC<-ymd_hms(df$CREATED)

# considerando que las variables climaticas presentan comporamientos ciclicos
# en periodos de 24 horas (dia-noche) y de 365 dias (por estaciones del año)
# es que se hace descomposicion del momento de creacion del registro en dos columnas
# una solo contiene la hora del registro
df2$onlyTime<-as_hms(df2$dateUTC)
# y otra solo el dia
df2$onlyDate<-date(df2$dateUTC)
head(df2)
summary(df2)

# mantener solo aquellos registros creados desde que la variable 'serverDate' fue implementada
# de lo contrario omitir esta linea
df2<-df2[!(df2$dateUTC<=ymd_hms("2021-06-25 16:45:25")),]
# df2<-df2[!(df2$CREATED<=ymd_hms("2021-06-24 16:45:25")),]
head(df2)

### Visualización de datos
## temperatura
# Evolucion temperatura a traves de lo que va del año
ggplot(df2,aes(x=dateUTC,y=AMBIENT_TEMPERATURE)) + geom_line() + 
  ggtitle("Evolución de temperatura") + 
  xlab("Momento") + 
  ylab("Temperatura")
# Evolucion temperatura cada 24 horas
ggplot(df2,aes(x=dateUTC,y=AMBIENT_TEMPERATURE,color=onlyTime)) + geom_point() + 
  ggtitle("Distribución de temperatura, color=momento de registro") + 
  xlab("Hora") + 
  ylab("Temperatura")
# Distribución temperatura cada 24 horas, clasificadas por día
ggplot(df2,aes(x=onlyTime,y=AMBIENT_TEMPERATURE, color=as.factor(onlyDate))) + geom_line() + 
  ggtitle("Distribución de temperatura cada 24 horas") + 
  xlab("Hora") + 
  ylab("Temperatura")

# Antiguamente hacia zoom a los registros pues en su mayoria existian pasadas las 14:00
# como esto ya no ocurre, esta seccion queda obsoleta
# df3<-df2
# df3<-df3 %>% 
#   filter(onlyTime > as_hms("14:00:00"))
# ggplot(df3,aes(x=onlyTime,y=HUMIDITY,color=as.factor(onlyDate))) + geom_line() + 
#   ggtitle("Distribución de temperatura pasadas las 14:00 horas") + 
#   xlab("Hora") + 
#   ylab("Temperatura")

## Humedad relativa
# Evolucion de la humedad relativa a traves de lo que va del año
ggplot(df2,aes(x=dateUTC,y=HUMIDITY)) + geom_line() + 
  ggtitle("Evolución de humedad relativa") + 
  xlab("Momento") + 
  ylab("Humedad relativa")
# Evolucion humedad relativa cada 24 horas. clasificadas por día
ggplot(df2,aes(x=onlyTime,y=HUMIDITY,color=as.factor(onlyDate))) + geom_line() + 
  ggtitle("Evolución de humedad relativa cada 24 horas") + 
  xlab("Hora") + 
  ylab("Humedad relativa")

## Presión atmosférica
# Evolucion de la presión atmosférica a traves de lo que va del año
ggplot(df2,aes(x=dateUTC,y=AIR_PRESSURE)) + geom_line() + 
  ggtitle("Evolución de presión atmosférica durante el año") + 
  xlab("Momento") + 
  ylab("Presión")
# Evolucion de la presión atmosférica cada 24 horas, clasificadas por día
ggplot(df2,aes(x=onlyTime,y=AIR_PRESSURE, color=as.factor(onlyDate))) + geom_line() +
 ggtitle("Distribución de presión atmosférica cada 24 horas") +
 xlab("Hora") +
 ylab("Presión")

## Relación entre las variables
# Aunque no se cuente con el tiempo de los registros, se puede estudiar su relacion al graficarlos
# por ejemplo, si la distribucion de los registros dibujan una fila, o forman grupos
ggplot(df2,aes(x=AMBIENT_TEMPERATURE,y=HUMIDITY,color=AIR_PRESSURE)) + geom_point()
rm(onlyHours)
ggplot(df2,aes(x=AMBIENT_TEMPERATURE,y=HUMIDITY,color=onlyTime)) + geom_point()
ggplot(df2,aes(x=AMBIENT_TEMPERATURE,y=HUMIDITY,color=dateUTC)) + geom_point()


ggplot(df2,aes(x=AIR_PRESSURE,y=AMBIENT_TEMPERATURE,color=HUMIDITY)) + geom_point()


########################
### Exploracion DGAC ###
####################3###

# importando archivos
temp <- read.csv("200006_2021_Temperatura_.csv",header = TRUE, sep = ";")
hum <- read.csv("200006_2021_Humedad_.csv",header = TRUE, sep = ";")
qfe <- read.csv("200006_2021_PresionQFE_.csv",header = TRUE, sep = ";")
# Join de los 3 datasets segun momento del registro
dgac=merge(temp,hum,by="momento")
dgac=merge(dgac,qfe,by="momento")
dgac$utc<-dmy_hms(dgac$momento)
# transformacion de fechas
dgac$onlyTime<-as_hms(dgac$utc)
dgac$onlyDate<-date(dgac$utc)
# filtrado de variables resultantes del join
# Ts_Valor = temperatura
# HR_Valor = humedad relativa
# QFE_Valor = presion atmosferica
toKeep<-c("Ts_Valor","HR_Valor","QFE_Valor","utc","onlyTime","onlyDate")
dgac2<-dgac[toKeep]
# eliminacion de datasets temporales
rm(hum)
rm(qfe)
rm(temp)

# temperatura
ggplot(dgac,aes(x=utc,y=Ts_Valor)) + geom_line() + 
  ggtitle("Evolución de temperatura en el año") + 
  xlab("Mes") + 
  ylab("Temperatura")
ggplot(dgac,aes(x=onlyTime,y=Ts_Valor,color=onlyDate)) + geom_jitter() + 
  ggtitle("Distribución de temperatura en el día") + 
  xlab("Hora") + 
  ylab("Temperatura")
ggplot(dgac,aes(x=hour(onlyTime),y=Ts_Valor, alpha=0.5)) + geom_jitter() + 
  facet_wrap(month(dgac$utc)) +
  ggtitle("Distribución de temperatura a través de las horas, separada por meses") +
  xlab("Hora") + 
  ylab("Temperatura")

# humedad
ggplot(dgac,aes(x=utc,y=HR_Valor)) + geom_line() + 
  ggtitle("Evolución de humedad relativa en el año") + 
  xlab("Mes") + 
  ylab("Humedad")
ggplot(dgac,aes(x=onlyTime,y=HR_Valor,color=onlyDate)) + geom_jitter() + 
  ggtitle("Distribución de humedad relativa en el día") + 
  xlab("Hora") + 
  ylab("Humedad")
ggplot(dgac,aes(x=hour(onlyTime),y=HR_Valor, alpha=0.5)) + geom_jitter() + 
  facet_wrap(month(dgac$utc)) +
  ggtitle("Distribución de humedad relativa a través de las horas, separada por meses") +
  xlab("Hora") + 
  ylab("Humedad")

# presion
ggplot(dgac,aes(x=utc,y=QFE_Valor)) + geom_line() + 
  ggtitle("Evolución de presión atmosférica en el año") + 
  xlab("Mes") + 
  ylab("Presión")
ggplot(dgac,aes(x=onlyTime,y=QFE_Valor,color=onlyDate)) + geom_jitter() +
  ggtitle("Distribución de presión atmosférica en el día") + 
  xlab("Hora") + 
  ylab("Presión")
ggplot(dgac,aes(x=hour(onlyTime),y=QFE_Valor, alpha=0.5)) + geom_jitter() + 
  facet_wrap(month(dgac$utc)) +
  ggtitle("Distribución de presión atmosférica a través de las horas, separada por meses") +
  xlab("Hora") + 
  ylab("Presión")

# relacion entre las variables
ggplot(dgac,aes(x=Ts_Valor,y=HR_Valor, color=QFE_Valor)) + geom_point() + 
  ggtitle("Distribución de humedad relativa y presión atmosférica respecto a la temperatura") + 
  xlab("Temperatura") + 
  ylab("Humedad")
ggplot(dgac,aes(x=QFE_Valor,y=Ts_Valor,color=HR_Valor)) + geom_point() + 
  ggtitle("Distribución de temperatura y humedad relativa respecto a la presión atmosférica") + 
  xlab("Presion") + 
  ylab("Temperatura")
ggplot(dgac,aes(x=QFE_Valor,y=Ts_Valor,color=HR_Valor)) + geom_point() + 
  facet_wrap(month(dgac$utc)) +
  ggtitle("Distribución de temperatura y humedad relativa respecto a la presión atmosférica, evolución por meses") + 
  xlab("Presión") + 
  ylab("Temperatura")

