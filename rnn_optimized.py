# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict stock price"""

__author__ = 'Qiyun Lu'


# Section One: data preprocessing

import numpy as np
import pandas as pd

# import the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:4].to_numpy()

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# create a data structure with 120 timesteps and 1 output
timesteps = 120
x_train = []
y_train = []
for i in range(timesteps, training_set_scaled.shape[0]):
    x_train.append(training_set_scaled[i-timesteps:i, :])
    y_train.append(training_set_scaled[i, :])
x_train, y_train = np.array(x_train), np.array(y_train)


# Section Two: build the rnn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initialize the rnn
regressor = Sequential()

# add the first LSTM layer and do dropout regularization
output_units = 64
dropout_rate = 0.2
input_dim = 3  # three indicators: open price, highest price and lowest price
regressor.add(LSTM(output_units, return_sequences=True, input_shape=(timesteps, input_dim)))
regressor.add(Dropout(dropout_rate))
# the second LSTM and dropout layers
regressor.add(LSTM(output_units, return_sequences=True))
regressor.add(Dropout(dropout_rate))
# the third LSTM and dropout layers
regressor.add(LSTM(output_units, return_sequences=True))
regressor.add(Dropout(dropout_rate))
# the fourth (last) LSTM and dropout layers
regressor.add(LSTM(output_units, return_sequences=False))
regressor.add(Dropout(dropout_rate))

# add the output layer
regressor.add(Dense(input_dim))

# compile the rnn
# my computer is not strong enough to fit 'RMSProp' optimizer to this large amount of data
# so I use default optimizer 'Adam'
regressor.compile('adam', loss='mean_squared_error')

# fit the rnn to the training set
batch_size = 32
epochs = 128
regressor.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)


# Section Three: make the prediction and visualize the results

# get the real stock prices of Jan 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, [1]].to_numpy()

# predict the stock prices of Jan 2017
# load and preprocess the test set
dataset_total = pd.concat([dataset_train, dataset_test], axis=0)
inputs = dataset_total.iloc[dataset_train.shape[0]-timesteps:-1, 1:4].to_numpy()
inputs = sc.transform(inputs)
x_test = []
for i in range(timesteps, inputs.shape[0]+1):
    x_test.append(inputs[i-timesteps:i, :])
x_test = np.array(x_test)
# make prediction
predicted_stock_price = regressor.predict(x_test)
# inverse the scaled prices to original ones
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualize the results
import matplotlib.pyplot as plt
plt.plot(real_stock_price[:, [0]], color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price[:, [0]], color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# Section Four: parameter tuning

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].to_numpy()

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

timesteps = 120
x_train = []
y_train = []
for i in range(timesteps, training_set_scaled.shape[0]):
    x_train.append(training_set_scaled[i-timesteps:i, :])
    y_train.append(training_set_scaled[i, :])
x_train, y_train = np.array(x_train), np.array(y_train)

def build_regressor(output_units):

    regressor = Sequential()
    dropout_rate = 0.2
    input_dim = 1
    regressor.add(LSTM(output_units, return_sequences=True, input_shape=(timesteps, input_dim)))
    regressor.add(Dropout(dropout_rate))
    regressor.add(LSTM(output_units, return_sequences=True))
    regressor.add(Dropout(dropout_rate))
    regressor.add(LSTM(output_units, return_sequences=True))
    regressor.add(Dropout(dropout_rate))
    regressor.add(LSTM(output_units, return_sequences=False))
    regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(input_dim))
    regressor.compile('adam', loss='mean_squared_error')
    return regressor

classifier = KerasClassifier(build_fn=build_regressor)

# dictionary of parameters to test
parameters = {'batch_size': [16, 32],
              'epochs': [64, 128],
              'output_units': [32, 64]}

# fit to the training set
grid_search = GridSearchCV(classifier, parameters, scoring='neg_mean_squared_error', n_jobs=2, cv=3)
grid_search = grid_search.fit(x_train, y_train)
best_parameter = grid_search.best_params_
best_score = grid_search.best_score_
print('Best parameter:', best_parameter, '\nBest score:', best_score)

