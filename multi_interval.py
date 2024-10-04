#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 11:15:50 2022

@author: robinson
"""
       
# to remove sklearns annoying warnings about everything and their mom
import warnings

# brother files
import Visualize
import File_Utils

# plotting... yes we'll get it moved to the visualize file soon
import matplotlib.pyplot as plt

# models
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

# preprocessing
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np

##### parameters that would be helpful to tweak #########
crypto = 'ADA' # any of the 10 scraped ones
train_intervals = ['1m', '5m', '15m', '30m', '1h', '2h'] # intervals used for the selected features
features = ['EMA10', 'SMA10', 'EMA20', 'SMA20', 'EMA50', 'SMA50', 'EMA100', 'SMA100', \
  'EMA200', 'SMA200', 'UO', 'AO', 'ADX', 'RSI', 'Mom', 'HullMA', 'STOCH.K', 'VWMA', 'Stoch.RSI', \
      'BBP', 'MACD', 'Ichimoku', 'CCI', 'W%R', 'EMA30', 'SMA30'] # do not edit - keep as reference

# 120 look ahead [5, 15, 30, 1, 2] ~ 60% for ADA 1m slightly helps 
#features = ['MACD', 'Stoch.RSI', 'Mom', 'RSI', 'SMA100', 'BBP', 'Ichimoku', 'UO', 'EMA10', 'VWMA', 'W%R'] # tweak feaatures here

look_aheads = 90 # how many minutes ahead to look (keep below 180)

splits = 5 # k folds - if 10: means 9 sets are training data and 1 is test data (repeated 10 times)
shuffled = True # shuffle training data?
start_cb_num = 0 # start file_num
end_cb_num = 59 # end file_num

##########################################################

# not being tweaked currently
window = 1 # only used with RNN (num samples / batch) - not being used
bool_roi = True
numerical = True
time_mode = '1st'
price_comp = True
tern = False

# models beings used - feel free to tweak if you want
models = {'K Nearest Neighbors': KNeighborsClassifier(n_neighbors = 5), \
          'Support Vector Machine': SVC(kernel = 'rbf', gamma = 'scale', probability = False, C = 100), \
              'Random Forest': RandomForestClassifier(), 'Decision Tree': DecisionTreeClassifier(), \
                 'Naive Bayesian': GaussianNB(), 'Log Regression': LogisticRegression(solver = 'saga', penalty = 'elasticnet', l1_ratio = 1, max_iter=1000), \
                     'Ada Boost': AdaBoostClassifier(), 'MLP': MLPClassifier(max_iter=1000, alpha=.01, early_stopping=True, hidden_layer_sizes=(15, 6, 3)), \
                          'QDA': QuadraticDiscriminantAnalysis(), 'Ridge Classifier': RidgeClassifier()}

# gets the data
all_x, all_y, all_roi, all_numeric_x, cols = File_Utils.load_multi_training_data(start_cb_num, end_cb_num, time_mode, train_intervals, \
                                 crypto, features, look_aheads)


# alternative testing method below    
'''
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=.2, train_size=.8)

for key in models.keys():
    model = models[key].fit(x_train, y_train)
    print(key, round(model.score(x_test, y_test), 4))
'''    

# used to get sklearn to stop giving warnings on colinearity and poor precision. relevant for QDA/ Bayesian/ MLP
def warn(*args, **kwargs):
    pass
warnings.warn = warn 


# splits the data in order so ROIs match up
def custom_split(all_x, splits):
    length = all_x.shape[0] // splits
    rounded_length = length * splits
    all_index = {i for i in range(rounded_length)}
    for split_no in range(splits):
        test = {i for i in range(split_no * length, (split_no + 1) * length, 1)}
        train = all_index - test
        train = np.asarray(list(train))
        if shuffled:
            np.random.shuffle(train)
        yield train, list(test)
        
 # metrics to be used   
scoring = ['accuracy', 'precision', 'recall_macro', 'f1_macro']
print('====================================================================== \nfeatures: {0}'.\
      format(features))
for key in models.keys():
    # custom split can be replaced with splits, but splits will be stratisfied and random  
    evals = cross_validate(models[key], all_x, all_y, cv = custom_split(all_x, splits), \
                                return_estimator = True, scoring = scoring)
    Visualize.print_training_metrics(evals, key)
    bot_list = [] # list of total money across all folds for a single bot
    wrongs = [] # list of wrong predictions for a bot - not currently being displayed/ used
    hold_plot = [999] # money list if just holding
    bot_plot = [1000] # list of money for a bot
    hold = 999 # starting money includes purchase fee
    bot = 1000 # starting money
    
    for estimator_num in range(splits):
        fitter = evals['estimator'][estimator_num] # model instance used for prediciton
        invested = False # if money is invested or not
        wrong = 0 # number of wrong predictions - not being used currrently
        for count in range(int(estimator_num * len(all_roi) / splits), int((estimator_num + 1) * len(all_roi) / splits), 1):
            hold += all_roi[count] * hold # holding amount of money
            
            # adds up money for the bot based if its invested or not
            if fitter.predict([all_x[count]]):
                if not invested:
                    bot -= bot * .001
                    invested = True
                bot += all_roi[count] * bot
                if all_roi[count] <= 0:
                    wrong += 1 
            else:
                if invested:
                    bot -= bot * .001
                    invested = False
                if all_roi[count] > 0:
                    wrong += 1
            hold_plot.append(hold)
            bot_plot.append(bot)
        
        wrongs.append(wrong)
        bot_list.append(round(bot, 2))
    # displays and make graphs
    money_time = round((60 * (sum(bot_plot) - sum(hold_plot)) / (look_aheads * len(bot_plot))), 2)
    print('>>> hold ending money: $', round(hold, 2), ' >>> bot ending money: $', bot_list[-1])
    
    plt.plot(bot_plot, label=key + ' ' + str(money_time) + ' USD/hr')
    if key.startswith('Supp'):
        plt.plot(hold_plot, linestyle=':', label='holding: $' + str(round(hold_plot[-1], 2)))
    plt.legend()
    plt.suptitle('Money ($) vs Time (' + str(look_aheads) + ' min increments)')
    plt.show()        
           
     
       
       