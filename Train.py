#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:31:37 2021

@author: robinson
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier, Lasso
# For model testing
from sklearn import metrics
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold

# misc
import datetime
import random
import numpy as np
import File_Utils
import Visualize
markov_number = 1
prob = False
num_loops = 1
display = False
rois_dic = {}
look_aheads = 120
rois_dic['holding'] = 0

time_mode = 'Avg' # Avg or 1st or 2nd (ie get trading view data at both 6s and 36s)
cb_num = 33
file_num = 0 # highest file number to load
k_folds = 2 # num of cross validation folds
train_interval = '1m' # time interval
crypto = 'ADA' # crypto name
# preloaded models and their names
models = {'K Nearest Neighbors': KNeighborsClassifier(n_neighbors = 5), \
          'Support Vector Machine': SVC(kernel = 'rbf', gamma = 'scale', probability = False, C = 100), \
              'Random Forest': RandomForestClassifier(), 'Decision Tree': DecisionTreeClassifier(), \
                 'Naive Bayesian': GaussianNB(), 'Log Regression': LogisticRegression(solver = 'saga', penalty = 'elasticnet', l1_ratio = 1), \
                     'Ada Boost': AdaBoostClassifier(), 'MLP': MLPClassifier(alpha = .01, max_iter = 1000, solver = 'adam', learning_rate_init=.01, hidden_layer_sizes = (26, 13, 7, 3), learning_rate = 'invscaling', early_stopping=True), \
                          'QDA': QuadraticDiscriminantAnalysis(), 'Categorical Bayesian': CategoricalNB(min_categories=5), 'Ridge Classifier': RidgeClassifier()}

for model_key in models.keys():
    rois_dic[model_key] = 0    
models2save = ['Support Vector Machine'] #  select models to save

# all:
features = ['EMA10', 'SMA10', 'EMA20', 'SMA20', 'EMA50', 'SMA50', 'EMA100', 'SMA100', \
 'EMA200', 'SMA200', 'UO', 'AO', 'ADX', 'RSI', 'Mom', 'HullMA', 'STOCH.K', 'VWMA', 'Stoch.RSI', \
     'BBP', 'MACD', 'Ichimoku', 'CCI', 'W%R', 'EMA30', 'SMA30']

#features = ['MACD', 'RSI', 'Mom']
#features = ['Mom']

#features = ['BBP', 'HullMA', 'SMA20']
# 30m ada SVM
#features = ['EMA20', 'EMA50', 'SMA50', 'SMA200', 'EMA30', 'HullMA', 'VWMA', 'Mom']
#15m ada ada boost
#features = ['EMA100', 'SMA100', 'STOCH.K', 'BBP', 'HullMA']
# 5m ada (2 look aheads) tree forest svm
#features = ['HullMA', 'Stoch.RSI', 'MACD', "AO"]
# 2h naive baye ada
#features = ['EMA100', 'EMA200', 'Mom', 'HullMA',  'VWMA', 'BBP', 'MACD', 'CCI', 'W%R', 'EMA30', 'SMA30', 'EMA10', 'SMA10', 'SMA20', 'SMA50']
# 1h k nearest ada
#features = ['SMA10', 'SMA30', 'EMA30']
#features = ['EMA200', 'Mom', 'HullMA',  'VWMA', 'BBP', 'MACD', 'CCI', 'W%R', 'EMA30', 'SMA30']
def train(file_num, time_mode, train_interval, crypto, features, rois_dic, display, num_loops):
    
    all_x, all_y, all_roi, all_numeric_x, _ = \
        File_Utils.load_training_data(time_mode, train_interval, \
                                      crypto, features, file_num, look_aheads, markov_number, cb_num)
    
    calc_metrics(models, k_folds, features, all_x, all_y) 
    for count in range(num_loops):
        rois_dic = calc_rois(models, all_x, all_y, all_roi, k_folds, features, \
                             train_interval, rois_dic, display, prob)
    for roi_key in rois_dic:
        rois_dic[roi_key] = round(rois_dic[roi_key], 4)
    print(rois_dic)
    print('Over {0} {1} intervals'.format(num_loops * look_aheads * (all_y.shape[0] // k_folds), train_interval))
    fit_models_for_saving(models2save, models, all_x, all_y, crypto, train_interval)
    

def calc_metrics(models, k_folds, features, all_x, all_y):
    ''' uses (stratified) k cross fold validation to evaluate each model '''
    scoring = ['accuracy', 'precision', 'recall_macro', 'f1_macro']
    print('====================================================================== \nfeatures: {0}'.\
          format(features))
    for key in models.keys():
       if key.startswith('Cat'):
               all_x = all_x * 2
               for row in range(all_x.shape[0]):
                   for col in range(all_x.shape[1]):
                       if all_x[row][col] < 0:
                           all_x[row][col] = all_x[row][col] + 5
           
       evals = cross_validate(models[key], all_x, all_y, cv = k_folds, \
                                    return_estimator = True, scoring = scoring)
       Visualize.print_training_metrics(evals, key)


def calc_rois(models, all_x, all_y, all_roi, k_folds, features, train_interval, rois_dic, display, prob):
    rnd_state = random.randint(0, 10000000)
    if display:
        print('====================================================================== \nfeatures: {0}'.\
              format(features))
    here = False
    for key in models.keys():
       holding_rois = []
       rois = []
       sum_holding_roi = 0
       sum_roi = 0
       kf = KFold(n_splits = k_folds, shuffle = True, random_state = rnd_state)
       
       for train_index, test_index in kf.split(all_x, all_y):
           
           
           holding_roi = 0
           roi = 0
           x_train, x_test = all_x[train_index], all_x[test_index]
           y_train, y_test = all_y[train_index], all_y[test_index]
           roi_train, _ = all_roi[train_index], all_roi[test_index]
           if key.startswith('Cat'):
               x_train = x_train * 2
               x_test = x_test * 2
               # should use sklearn preprocessing ordinal encoder
               for row in range(x_train.shape[0]):
                   for col in range(x_train.shape[1]):
                       if x_train[row][col] < 0:
                           x_train[row][col] = (x_train[row][col]) + 5
                   
               for row in range(x_test.shape[0]):
                   for col in range(x_test.shape[1]):
                       if x_test[row][col] < 0:
                           x_test[row][col] = (x_test[row][col]) + 5
           elif key.startswith('FIX'):
               
               max_ = max(roi_train)
               min_ = min(roi_train)
               range_ = max_ - min_
               np.add(roi_train, [-1 * min_], out=roi_train, where=True)
               
               roi_train /= range_
               
               temp = y_train 
               y_train = roi_train
               here = True
           elif here:
               y_train = temp
                
    
           roi_test = all_roi[test_index]
           new_model = models[key].fit(x_train, y_train)
           
           if prob:
               y_pred = new_model.predict_proba(x_test)
           else:
               y_pred = new_model.predict(x_test)
           
           y_pred = np.reshape(y_pred, (-1))
           for idx in range(len(test_index)):
               if prob:
                   if y_pred[idx] > .5:
                        y_pred[idx] = True
                   else:
                        y_pred[idx] = False
               if (y_test[idx] and y_pred[idx]) or (not y_test[idx] and (not y_pred[idx])):
                   roi += max(0, roi_test[idx])
               else:
                   roi += min(0, roi_test[idx])
               holding_roi += roi_test[idx]
           rois.append(round(roi, 4))
           sum_roi += roi
           holding_rois.append(round(holding_roi, 4)) 
           sum_holding_roi += holding_roi
       
       avg_roi = round(sum_roi / k_folds, 4)
       rois_dic[key] += avg_roi
       avg_holding_roi = round(sum_holding_roi / k_folds, 4)
       if display:
           Visualize.print_training_rois(key, rois, avg_roi, \
                                         holding_rois, avg_holding_roi)
    if display:
        print('''
                  Over {0} {1} intervals'''.format(len(test_index), train_interval))
    rois_dic['holding'] += avg_holding_roi
    return rois_dic
          
def fit_models_for_saving(models2save, models, all_x, all_y, crypto, train_interval):
    thyme = datetime.datetime.now().strftime("%D_%H_%M")
    for name in models2save:
        to_save = models[name].fit(all_x, all_y)
        File_Utils.save_models(to_save, features, crypto, train_interval, thyme, name)

train(file_num, time_mode, train_interval, crypto, features, rois_dic, display, num_loops)

# 15 m ADA naive bayesian 2.25% for 29 hrs
features = ['UO', 'RSI', 'Stoch.RSI', 'AO', 'Mom', 'HullMA', 'BBP', 'EMA10', 'SMA10', 'VWMA']


# 30 m DOT Decision Tree 7.9% 79hr
features = ['Mom', 'EMA10', 'HullMA', 'SMA10', 'EMA20', 'VWMA', 'SMA20', 'EMA30', 'SMA30', 'MACD', 'EMA50', 'SMA50']

#30 m DOT naive baye
features = ['EMA200', 'SMA200', 'BBP']

# 15 m DOT 2 for look ahead random forest 11% 228hrs
features = ['EMA200', 'SMA200', 'EMA100', 'MACD', 'RSI']

#features = ['BBP', 'EMA200', 'SMA200']
#features = ['HullMA', 'AO', 'Mom']

#1inch
# SVM 10 10 - 7% (hodl -.88%) 1inch 30m 375hr not good????
# features = ['HullMA', 'BBP', 'EMA50', 'SMA50', 'SMA200']

# ada boost 10 10 - 9% (hodl -.88%) 1inch 30m 375 hr
#features = ['HullMA', 'BBP', 'EMA50', 'SMA50', 'SMA200', 'EMA100', 'SMA100', 'EMA200']


# MLP 22% 375 hr 1inch 15m 5 5 hodl -3%
#features = ['EMA10', "SMA10", 'SMA20', "BBP", 'CCI', 'AO', 'W%R', "VWMA"]

# LOG 22% 375 hr 1inch 15m 5 5 hodl -3%
#features = ['EMA10', 'CCI', 'BBP', 'AO', 'W%R', 'Stoch.RSI']

# 1inch 5 min 385 hr 5 5 25% MLP or LOG and -9 % hold
features = ['CCI', 'W%R', 'Stoch.RSI', 'STOCH.K', 'HullMA', 'MACD']
# same as above but look ahead 2
features = ['CCI', 'AO', 'MACD', 'W%R', 'VWMA', 'STOCH.K', 'RSI']

# 1 inch 180 hr 7% hold -12% tree/ forest 1 hr
#features = ['MACD', 'Mom', 'BBP', 'SMA100']
# comparable to above
#features = ['Mom', 'BBP', 'SMA100', 'EMA50']







# trading view is not markov - use past states too - averaging to start maybe