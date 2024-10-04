#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 15:09:06 2022

@author: robinson
"""

###### works pretty bad #####


import File_Utils
import Format
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, Embedding
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
window = 24
cb_num = 20
file_num = 36
look_aheads = 1
markov_number = 1
bool_roi = True
numerical = False
time_mode = '1st'
interval = '5m'
crypto = 'ADA'
tern=False
splits = 10
features = ['EMA10', 'SMA10', 'EMA20', 'SMA20', 'EMA50', 'SMA50', 'EMA100', 'SMA100', \
  'EMA200', 'SMA200', 'UO', 'AO', 'ADX', 'RSI', 'Mom', 'HullMA', 'STOCH.K', 'VWMA', 'Stoch.RSI', \
      'BBP', 'MACD', 'Ichimoku', 'CCI', 'W%R', 'EMA30', 'SMA30']
 
features = ['Mom']

all_x, all_y, all_roi, all_numeric_x, cols = File_Utils.load_training_data(time_mode, \
                                                      interval, crypto, features, \
                                                        file_num, look_aheads, \
                                                            markov_number, cb_num, window, tern)
'''    
# add prices into x (change len(features) + 1)
all_roi = all_roi[:-1] * 100   
all_roi = np.insert(all_roi, 0, 0, axis=0)
all_x = np.array(all_x, dtype=float)
all_x = np.insert(all_x, 0, all_roi, axis=1)
'''

all_x = np.reshape(all_x, (-1, window, len(features)))

mode = ''

# 1 logit use top 2 lines - multi logit use bottom and set units=window below
all_y = all_y[window - 1::window]
all_roi = all_roi[window - 1::window]
#all_y = np.reshape(all_y, (-1, window))
#all_roi = np.reshape(all_roi, (-1, window))

# just prices as feature
#all_x = np.reshape(all_roi, (-1, window, 1))

def build_model(window):
  
    window_size = window
    model = Sequential()
    
    #First recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size*3, return_sequences=True), input_shape=(window_size, X_train.shape[-1]),))
    model.add(Dropout(.8))
    
    #Second recurrent layer with dropout
    model.add(Bidirectional(LSTM((window_size*6), return_sequences=True)))
    model.add(Dropout(.5))
    
    #Third recurrent layer
    model.add(Bidirectional(LSTM(window_size, return_sequences=False)))
    
    #Output layer (returns the predicted value)
    model.add(Dense(units=1, kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-5),
    bias_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
    
    #Set activation function
    model.add(Activation('sigmoid'))
    
    #Set loss function and optimizer
    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics='Accuracy')
    return model

def dual_model(window):
  
    window_size = window
    model = Sequential()
    
    #First recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size*3, return_sequences=False), input_shape=(window_size, X_train.shape[-1]),))
    model.add(Dropout(.8))
    
    
    
    model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
    bias_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4)))
    
    model.add(Dropout(.5))
    
    #Output layer (returns the predicted value)
    model.add(Dense(units=1))
    
    #Set activation function
    model.add(Activation('sigmoid'))
    
    #Set loss function and optimizer
    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics='Accuracy')
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics='mean_absolute_percentage_error')
    return model

def multi_class(window):
    window_size = window
    model = Sequential()
    
    #First recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size * 10, return_sequences=False), input_shape=(window_size, X_train.shape[-1]),))
    model.add(Dropout(.8))
    
    model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
    bias_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4)))
    
    model.add(Dropout(.5))
    
    #Output layer (returns the predicted value)
    model.add(Dense(units=3))
    
    #Set activation function
    model.add(Activation('softmax'))
    
    #Set loss function and optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='Accuracy')
    return model



def train(model):
    hist = model.fit(X_train, y_train, batch_size=64, epochs=30)
    epochs = hist.history['loss']
    
   # plt.plot(epochs)
    
    return model, epochs


def roi_trainer(roi_train, roi_test):
    model = dual_model(window)
    max_ = max(roi_train)
    min_ = min(roi_train)
    range_ = max_ - min_
    np.add(roi_train, [-1 * min_], out=roi_train, where=True)
    np.add(roi_test, [-1 * min_], out=roi_test, where=True)
    roi_train /= range_
    roi_test /= range_
    assignment = lambda l: 1 if l > 1 else (0 if l < 0 else l)
    roi_test = np.asarray(list(map(assignment, roi_test)))
    
    
    hist = model.fit(X_train, roi_train, batch_size=64, epochs=3)
    epochs = hist.history['loss']
    
    plt.plot(epochs)
    
    return model, epochs, roi_test

def test(model, y_test):
    score = model.evaluate(X_test, y_test)
   
    return score

kf = KFold(n_splits=splits)

scores = []
KFold(n_splits=splits, random_state=None, shuffle=False)
for train_index, test_index in kf.split(all_x):

    X_train, X_test = all_x[train_index], all_x[test_index]
    y_train, y_test = all_y[train_index], all_y[test_index]
    roi_train, roi_test = all_roi[train_index], all_roi[test_index]
    if tern:
       model = multi_class(window)
    else:
        if mode!= 'roi':
            #model = build_model(window)
            model = dual_model(window)
        else:
            model, epochs, alt_y_test = roi_trainer(roi_train, roi_test)
    if mode != 'roi':
        model, epochs = train(model)
    score = test(model, y_test)
    scores.append(score[1])
print('Accuracies: ', np.round(scores, 4))
print('Macro accuracy:', round(np.mean(np.asarray(scores)), 4))


    
#### data is biased towards bear since binary classifier is either up or not up 
# time to finally try ternary classifier?????

# sparse categorical cross entropy
# kullback-liebler
