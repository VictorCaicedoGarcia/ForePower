# -*- coding: utf-8 -*-
"""forecasting.ipynb

# Condiciones de los datos de entrada:
Debe ser un archivo .csv con 3 columnas:
    1.primera columna indica la fecha de los datos
        Titulo = 'Date'
        Forma de los datos: Año-Mes-Dia horas:minutos:segundos
            Ejemplo: 2019-01-01 0:00:00
    2. Segunda Columna: Carga en MW, tipo número, Título ='Load'
    3. Tercera Columna: Temperatura en Grados Celsius, Tipo número, Título='Temperature'

"""

# If using data in Google drive
    # Access to .csv in drive

    #from google.colab import drive
    #drive.mount('/content/drive')

# Import modules and packages
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime as dt

from datetime import datetime
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

%matplotlib inline

#Insert .csv file
df = pd.read_csv('/content/drive/Shared drives/ISE Compartido/Forecasting/RNN/DateTimeFine.csv',index_col = 0, parse_dates = [0])

#Function to plot
titles = ["Load","Temperature"]
feature_keys = ["Load", "Temperature"]
colors = ["blue","orange"]

def show_raw_visualization(data):
    time_data = data.index
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 1],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()

#First data view

#plot All data
show_raw_visualization(df)
plt.savefig('Actual Year Load.png')

#plot Day data
show_raw_visualization(df.iloc[0:24,:])
plt.savefig('Actual Day Load.png')

#plot Week data
show_raw_visualization(df.iloc[0:24*7,:])
plt.savefig('Actual Week Load.png')

#plot Month January data
show_raw_visualization(df.iloc[0:24*7*31,:])
plt.savefig('Actual January Load.png')

!cp Actual\ Year\ Load.png
!cp Actual\ Day\ Load.png
!cp Actual\ January\ Load.png
!cp Actual\ Week\ Load.png

#Correlation plot
def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)

    plt.show()

show_heatmap(df)

plt.savefig('CorrelationHeatmap.png')
!cp CorrelationHeatmap.png

# Select features (columns) to be involved intro training and predictions
cols = list(df)[0:2]

# Extract dates (will be used in visualization)
datelist_train = list(df.index)
print('Training set shape == {}'.format(df.shape))
print('All timestamps == {}'.format(len(df.index)))
print('Featured selected: {}'.format(cols))

dataset_train = df
dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)

# Using multiple features (predictors)
training_set = dataset_train.values

print('Shape of training set == {}.'.format(training_set.shape))
training_set

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:, 0:1])

# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

#Default predicted_days
predicted_days = 7
""" INSERT LINE TO INSERT NUMBER OF DAYS TO PREDICT """
n_future = 24*predicted_days   # Number of days we want top predict into the future
n_past = 24*31     # Number of past days we want to use to predict the future


for i in range(n_past, len(training_set_scaled) - n_future +1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))

"""#2. Create Model
## Training: Building the LSTM based Neural Network
"""

# Import Libraries and packages from Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam

# Initializing the Neural Network based on LSTM
model = Sequential()

# Adding 1st LSTM layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))

# Adding 2nd LSTM layer
model.add(LSTM(units=10, return_sequences=False))

# Adding Dropout
model.add(Dropout(0.25))

# Output layer
model.add(Dense(units=1, activation='linear'))

# Compiling the Neural Network
model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')

"""## Start Training"""

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")
plt.savefig('Training and Validation Loss.png')
!cp Training\ and\ Validation\ Loss.png  "/content/drive/Shared drives/ISE Compartido/Forecasting/RNN"

"""#3. Make Future predictions"""

# Generate list of sequence of days for predictions
datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
#I have to create a new datelist_future

datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())

df2020 = pd.read_csv('/content/drive/Shared drives/ISE Compartido/Forecasting/RNN/Date2020.csv',index_col = 0, parse_dates = [0])

dfReal2020 = pd.read_csv('/content/drive/Shared drives/ISE Compartido/Forecasting/RNN/emicReal2020fine.csv',index_col = 0, parse_dates = [0])

datelist_future_hour= list(df2020.index[0:168])
type(datelist_future_hour), type(datelist_future)
#type(dfReal2020)

"""## 5. Make predictions for future dates"""

# Perform predictions
predictions_future = model.predict(X_train[-n_future:])
predictions_train = model.predict(X_train[n_past:])

# Inverse the predictions to original measurements

# ---> Special function: convert <datetime.date> to <Timestamp>
def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')

y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)


PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Load']).set_index(pd.Series(datelist_future_hour))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Load']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

"""## 6. Visualize data

###Whole Data
"""

# Set plot size
from pylab import rcParams
rcParams['figure.figsize'] = 20,8

# Plot parameters
START_DATE_FOR_PLOTTING = '2019-01-01'

plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Load'], color='r', label='Predicted Load')
plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')

plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
plt.title('Predictions and Actual Load', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.show()
plt.savefig('Actual vs predicted load.jpg')
!cp Actual\ vs\ predicted\ load.jpg

"""### Just Prediction"""

rcParams['figure.figsize'] = 20, 8

# Plot parameters
START_DATE_FOR_PLOTTING = '2020-01-01'

plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Load'], color='r', label='Predicted Load')
#plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
#plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')
plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
plt.title('Predictions Load', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.show()
plt.savefig('Predicted Load.jpg')
!cp Predicted\ Load.jpg

"""### December 2019 + 2020"""

rcParams['figure.figsize'] = 20, 8

# Plot parameters
START_DATE_FOR_PLOTTING = '2019-12-01'

plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Load'], color='r', label='Predicted Load')
#plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')
plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
plt.title('Predictions Load', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.show()
plt.savefig('December2019andJanuary2020.jpg')
!cp December2019andJanuary2020.jpg

#Compare real with prediction

rcParams['figure.figsize'] = 20, 8

# Plot parameters
START_DATE_FOR_PLOTTING = '2020-01-01'

plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Load'], color='r', label='Predicted Load')
plt.plot(dfReal2020.index[:168],dfReal2020['Load'][:168],color='orange',label ='Real 2020 Load' )
#plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
#plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')
plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
plt.title('Predictions Load', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.show()
plt.savefig('Realvspredicted.png')
!cp Realvspredicted.png

#Compare real with prediction

rcParams['figure.figsize'] = 20, 8

# Plot parameters
START_DATE_FOR_PLOTTING = '2020-01-03'

plt.plot(PREDICTIONS_FUTURE.index[48:73], PREDICTIONS_FUTURE['Load'][48:73], color='r', label='Predicted Load')
plt.plot(dfReal2020.index[48:73],dfReal2020['Load'][48:73],color='orange',label ='Real 2020 Load' )
#plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
#plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')
plt.axvline(x = min(PREDICTIONS_FUTURE.index[48:73]), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
plt.title('Predictions Load', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.show()
plt.savefig('Realvspredicted3January.png')
!cp Realvspredicted3January.png

"""## Preparing forecasting.csv"""

forecasting = PREDICTIONS_FUTURE
forecasting.to_csv('forecasting.csv')
!cp forecasting.csv "/content/drive/Shared drives/ISE Compartido/Forecasting/RNN"
