#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 21:04:26 2022

@author: robinson
"""

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
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



window = 1
cb_num = 30
file_num = 36
look_aheads = 1
markov_number = 1
bool_roi = True
numerical = True
time_mode = '1st'
interval = '15m'
crypto = 'ADA'
price_comp = True
tern = False
splits = 10
features = ['EMA10', 'SMA10', 'EMA20', 'SMA20', 'EMA50', 'SMA50', 'EMA100', 'SMA100', \
  'EMA200', 'SMA200', 'UO', 'AO', 'ADX', 'RSI', 'Mom', 'HullMA', 'STOCH.K', 'VWMA', 'Stoch.RSI', \
      'BBP', 'MACD', 'Ichimoku', 'CCI', 'W%R', 'EMA30', 'SMA30']
 
features = ['Mom']

models = {'K Nearest Neighbors': KNeighborsClassifier(n_neighbors = 5), \
          'Support Vector Machine': SVC(kernel = 'rbf', gamma = 'scale', probability = False, C = 100), \
              'Random Forest': RandomForestClassifier(), 'Decision Tree': DecisionTreeClassifier(), \
                 'Naive Bayesian': GaussianNB(), 'Log Regression': LogisticRegression(solver = 'saga', penalty = 'elasticnet', l1_ratio = 1, max_iter=1000), \
                     'Ada Boost': AdaBoostClassifier(), 'MLP': MLPClassifier(max_iter=1000, alpha=.1, early_stopping=True, hidden_layer_sizes=(40, 20, 10, 5, 2)), \
                          'QDA': QuadraticDiscriminantAnalysis()}


all_x, all_y, all_roi, all_numeric_x, cols = File_Utils.load_training_data(time_mode, \
                                                      interval, crypto, features, \
                                                        file_num, look_aheads, \
                                                            markov_number, cb_num, window, tern, price_comp)



piv = []
for i in range(51, 82):
    piv.append(i)

# use top, or bottom 2
# redundant now with subtraction: 4 7 8 10 14 15 19 21 86

all_numeric_x = np.delete(all_numeric_x, [4, 7, 8, 10, 14, 15, 19, 21, 86, 22, 24, 28, 30, 45, 47, 49, 82, 87, 88, 89, 90] + piv, 1) # 47, 49 ind rec and/or 0, 1, 2 overall rec
#remov = np.add(np.arange(88), [3] * 88)
#all_numeric_x = np.delete(all_numeric_x, remov, 1)

x_train, x_test, y_train, y_test = train_test_split(all_numeric_x, all_y, test_size=.2, train_size=.8)


x_train, x_test, roi_train, roi_test = train_test_split(all_numeric_x, all_roi, test_size=.2, train_size=.8)
func = lambda l: 1 if l > 1 else (l if l > -1 else -1)
for col_num in range(x_train.shape[1]):
    mx = x_train[:, col_num].max()
    mn = x_train[:, col_num].min()
    rng = mx - mn
  
    x_train[:, col_num] = np.add(x_train[:, col_num], [-1 * mn] * x_train.shape[0]) / (.5 * rng)
    x_test[:, col_num] = np.add(x_test[:, col_num], [-1 * mn] * x_test.shape[0]) / (.5 * rng)
    x_train[:, col_num] = np.add(x_train[:, col_num], [-1] * x_train.shape[0])
    x_test[:, col_num] = np.add(x_test[:, col_num], [-1] * x_test.shape[0])
    x_test[:, col_num] = list(map(func, x_test[:, col_num]))

for key in models.keys():
    model = models[key].fit(x_train, y_train)
    print(key, round(model.score(x_test, y_test), 4))


def build_model():
  

    model = Sequential()
    
    model.add(Dense(units=81, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
    bias_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
    
    model.add(Dense(units=40, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
    bias_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
    
    #Output layer (returns the predicted value)
    model.add(Dense(units=1))
    
    
    #Set loss function and optimizer
    model.compile(loss='mean_squared_error', optimizer='adam', metrics='mean_absolute_error')
    return model

def train(model):
    hist = model.fit(x_train, roi_train, batch_size=32, epochs=100)
    epochs = hist.history['loss']
    
   # plt.plot(epochs)
    
    return model, epochs


all_x = np.reshape(all_x, (-1, window, len(features)))

all_y = all_y[window - 1::window]
all_roi = all_roi[window - 1::window]

model = build_model()
model, epochs = train(model)
print(epochs[-1])
print(model.evaluate(x_test, roi_test))

# try rnn next
# or how about multiple intervals together like 52 features -  the 15 min and the 2hr features (use altCB)
