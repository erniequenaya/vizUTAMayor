#!/root/prediccionService/venv/bin/python

import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf

## Cargando set de datos
df = pandas.read_csv("../dataPreprocessed.csv")
dates_df = pandas.to_datetime(df.pop('utc'), format='%Y-%m-%d %H:%M:%S')

## Normalizando
df2 = df.copy()
train_mean = df.mean()
train_std = df.std()
df = (df - train_mean) / train_std

## "Codificando" tiempo como secuencia
timestamp_s = dates_df.map(pandas.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day
timeEncode = pandas.DataFrame()
timeEncode['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
timeEncode['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))

df['dsin']=timeEncode['Day sin']
df['ysin']=timeEncode['Year sin']

## Dividiendo dataset
n = len(df)
#train_df = df[0:int(n*0.7)]
train_df = df[0:int(n*0.8)]
#val_df = df[int(n*0.7):int(n*0.9)]
val_df = df[int(n*0.8):int(n*0.95)]
#test_df = df[int(n*0.9):]
test_df = df[int(n*0.95):]

num_features = df.shape[1]
num_features

## Creando sistema de ventanas, proveido por tutorial tensorflow [https://www.tensorflow.org/tutorials/structured_data/time_series]
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result


## Testeando sistema de ventanas
winSize=24
winLabel = 24
wide_window = WindowGenerator(
    #input_width=winSize, label_width=1, shift=1, # autoregresivo puro
    input_width=winSize, label_width=winLabel, shift=24, # ventanas de 24 horas
    label_columns=['Ts_Valor','HR_Valor','QFE_Valor','dsin','ysin'])
    #label_columns=['Ts_Valor','HR_Valor','QFE_Valor','ysin','ycos'])


wide_window

for example_inputs, example_labels in wide_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')


## Implementando algoritmo transformer adoptado a series de tiempo, proveido por keras 
# [https://keras.io/examples/timeseries/timeseries_transformer_classification/]
from tensorflow import keras
from tensorflow.keras import layers

## Construyendo capa codificadora de modelo
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    #x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="leaky_relu")(x)
    x = layers.Dropout(dropout)(x)
    #x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="leaky_relu")(x)
    return x + res

## Construyendo resto del modelo 
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    #x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Flatten()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    #outputs = layers.Dense(n_classes, activation="softmax")(x)
    outputs = layers.Dense(winLabel*num_features,activation="leaky_relu")(x)
    return tf.keras.Model(inputs, outputs)


## Obteniendo dimensiones de entrada
input_shape = wide_window.example[0].shape[1:]
input_shape

## Construyendo modelo base
model = build_model(
    input_shape,
    head_size=96,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

## Creando, compilando y entrenando modelo final
trans = tf.keras.models.Sequential()
trans.add(model)
trans.add(tf.keras.layers.Dense(winLabel*num_features,activation="linear"))
trans.add(tf.keras.layers.Reshape([-1, num_features]))

MAX_EPOCHS=3
## Desde la version 2.5 en adelante de tensorflow, tf ya no provee optimizadores de forma independiente
## sino que lo hace a traves de Keras -que al fin y al cabo viene integrado en tf. De esta forma solo cambia la ruta para referencias los optimizadores
#trans.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
trans.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=[tf.keras.metrics.MeanAbsoluteError()])
trans.fit(wide_window.train, epochs=MAX_EPOCHS, validation_data=wide_window.val,shuffle=False)

## Testeando, transferir codigo a trnsfmPredictions
df3 = val_df[-winSize:]

df3 = test_df[300:324]
df3 = test_df[600:624]
stackPreds = pandas.DataFrame()
df3 = df3.to_numpy()
df3 = df3.reshape((-1,winSize,num_features))
df3.shape

#now = pandas.to_datetime("2021-04-28 16:00:00")
#now = pandas.Timestamp.timestamp(now)
x = trans.predict(df3)
a = trans.predict(x)
b = trans.predict(a)
x = np.hstack((x,a))
x = np.hstack((x,b))


## Como el modelo genera outputs del mismo tipo que su input, puede implementarse de forma semi-autoregresiva,
## pues en vez de iria hora->hora va dia->dia

#stackPreds = pandas.DataFrame(np.reshape(x,(winLabel,num_features)))
stackPreds = pandas.DataFrame(np.reshape(x,(72,num_features)))
stackPreds.columns=['Ts_Valor','HR_Valor','QFE_Valor','dsin','ysin']
#stackPreds.columns=['Ts_Valor','HR_Valor','QFE_Valor','dsin','ysin','ycos']
#stackPreds.columns=['Ts_Valor','HR_Valor','QFE_Valor','ysin','ycos']

test_df = test_df.reset_index(drop=True)
first = 600
last = first+72
test_sample = test_df[first:last]
test_sample = test_sample.reset_index(drop=True)

plt.plot(test_sample['Ts_Valor'], label='testLabels')
plt.plot(stackPreds['Ts_Valor'],label='stackPreds')
plt.legend()
plt.grid()
plt.show()

plt.plot(test_sample['HR_Valor'], label='testLabels')
plt.plot(stackPreds['HR_Valor'],label='stackPreds')
plt.legend()
plt.grid()
plt.show()

plt.plot(test_sample['QFE_Valor'], label='testLabels')
plt.plot(stackPreds['QFE_Valor'],label='stackPreds')
plt.legend()
plt.grid()
plt.show()

#plt.plot(test_df['dsin'][first:last], label='testLabels')
#plt.plot(stackPreds['dsin'],label='stackPreds')
#plt.legend()
#plt.grid()
#plt.show()

plt.plot(test_df['ysin'], label='testLabels')
plt.plot(stackPreds['ysin'],label='stackPreds')
plt.legend()
plt.grid()
plt.show()

from sklearn.metrics import mean_absolute_error
a = mean_absolute_error(stackPreds[['Ts_Valor','HR_Valor','QFE_Valor']],test_df[['Ts_Valor','HR_Valor','QFE_Valor']][first:last],multioutput='raw_values')
a

trans.save("../../../models/transformer")
trans2
# array([0.51061708, 0.61037691, 0.64611513]) 24->24 to 72 dsin+ysin leakyrelu