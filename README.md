# UTA Mayor
El presente proyecto corresponde a codigo necesario para generar el motor de predicciones de variables climáticas para el proyecto UTA Mayor

# Estructura del proyecto
El proyecto se estructura de la siguiente forma:
/
├── oldVisualization(25-6)
│   └── visualizacoinAntes-25-6.html
├── scripts
│   ├── dataPreprocessed.csv
│   ├── pass
│   ├── predict
│   │   ├── \*\*\* Predictions.py
│   ├── pres9-8
│   ├── pres9-8.ipynb
│   └── reTrain
│       ├── dataPreprocessing.py
│       ├── \*\*\* Retrain.py
├── visualizacionDatosR
│   ├── 200006_2021_ \*\*\* .csv
│   ├── soloCodigo.R
│   ├── visualization.html
│   └── weather.csv
└── visualization.Rmd

Leyenda:
+ oldVisualization  : corresponde al archivo presentado en la primera visualización de datos del proyecto
+ visualizacionR    : corresponde a los archivos de visualización presentados post-implementacion de la variable serverDate en la base de datos
  - *200006_2021_\*\*\*.csv*  : csv con datos DGAC a la fecha de presentacion
  - soloCodigo.R              : archivo ejecutable con opcion para conexion a DB, para obtener visualizacion de datos locales en cualquier momento
  - visualization.html        : archivo presentado un mes despues de la 1ra visualizacion
  - weather.csv               : respaldo de datos locales
+ scripts           : carpeta con archivos ejecutables importantes para el funcionamiento del sistema de pronosticos UTA Mayor
  - **pass**                  : archivo con credenciales para acceso a base de datos UTA Mayor, debe crearse si no existe
  - predict                   : carpeta con scripst de predicciones
    + **Predictions.py**        : ejecutables
  - **dataPreprocessed.csv**  : archivo 

# Rmd 
El documento presentado el 25 de junio corresponde a un .html generado por un archivo R Markdown (.Rmd)
Este es un tipo de documento fuente que permite la escritura de texto markdown a la vez que  código R para su trabajo en conjunto
El código R es compilado al momento de exportar el documento, ya sea como .html .docx o .pdf
Para instalar la extensión ejecutar
install.packages("rmarkdown")

# Estado DB
Desde la fecha de presentacion la cantidad de registros de la base de datos ha de haber cambiado
Esto afecta la precisión de los gráficos y textos presentados el 25 de junio por lo tanto es probable que la ejecucion del archivo .Rmd de errores
Por favor leer y ejecutar "soloCodigo.R" en su reemplazo

## Actualizacion 19-07
ATENCION: El archivo visualization.Rmd va a compilar  **SOLO** si la contraseña de la DB es proveida como texto plano en vez de
```
.rs.askForPassword
```


