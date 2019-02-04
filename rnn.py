# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict stock price"""

__author__ = 'Qiyun Lu'


# Section One: data preprocessing

import numpy as np
import pandas as pd

# import the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, [1]].to_numpy()

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# create a data structure with 60 timesteps and 1 output
timesteps = 60
x_train = []
y_train = []
for i in range(timesteps, training_set_scaled.shape[0]):
    x_train.append(training_set_scaled[i-timesteps:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Section Two: build the rnn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initialize the rnn
regressor = Sequential()

# add the first LSTM layer and do dropout regularization
output_units = 32
dropout_rate = 0.2
regressor.add(LSTM(output_units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(dropout_rate))
# the second LSTM and dropout layers
regressor.add(LSTM(output_units, return_sequences=False))
regressor.add(Dropout(dropout_rate))

# add the output layer
regressor.add(Dense(1))

# compile the rnn
regressor.compile('adam', loss='mean_squared_error')

# fit the rnn to the training set
batch_size = 16
epochs = 64
regressor.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)


# Section Three: make the prediction and visualize the results

import matplotlib.pyplot as plt

# get the real stock prices of Jan 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, [1]].to_numpy()

# predict the stock prices of Jan 2017
# load and preprocess the test set
dataset_total = pd.concat([dataset_train['Open'], dataset_test['Open']], axis=0)
inputs = dataset_total[dataset_train.shape[0]-timesteps:-1].to_numpy()
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []
for i in range(timesteps, inputs.shape[0]+1):
    x_test.append(inputs[i-timesteps:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# make prediction
predicted_stock_price = regressor.predict(x_test)
# inverse the scaled prices to original ones
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualize the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
