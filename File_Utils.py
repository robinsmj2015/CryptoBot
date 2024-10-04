#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 08:40:54 2021

@author: robinson
"""
import sys
sys.path.append('../')
import Format
import pickle
import os
import config
import numpy as np

def create(data_to_save, is_alt_df):
    if is_alt_df:
        save_to_file_name = 'alt_test_df.pkl'  
    else:
        save_to_file_name = 'test_df.pkl'
    file = open(config.get_file_paths() + save_to_file_name, 'wb')
    pickle.dump(data_to_save, file)
    file.close()
    return
    
def delete(file_name):
    input1 = input('Are you sure you would like to delete {} ?'.format(file_name))
    if input1.lower().startswith('y'):
        os.remove(config.get_file_paths() + file_name)

def load_models(model_files):
    models = []
    for file_name in model_files:
        file = open(config.get_file_paths() + file_name, 'rb')
        models.append(pickle.load(file))
        file.close()
    return models
    
def load_features(feature_files):
    model_features = []
    for file_name in feature_files:
        file = open(config.get_file_paths() + file_name, 'rb')
        model_features.append(pickle.load(file))
        file.close()
    
    return model_features

def load_training_data(time_mode, train_interval, crypto, features, file_num, look_aheads, markov_number, cb_num, window=None, tern=False, price_comp=False):
    file_names = []
    all_x = None
    all_y = None
    all_roi = None
    all_numeric_x = None
    first_time = True
    for num in range(0, file_num + 1, 1):
       file_names.append('BIdf' + str(num) + '.pkl')
    for num in range(0, cb_num + 1):
        file_names.append('CB' + str(num) + '.pkl')
    # for each file gets the oscillators, moving averages and start/ end prices
    for file_name in file_names:
        file = open(config.get_file_paths() + file_name, 'rb')
        train_df_dic = pickle.load(file)
        file.close()
        all_x, all_y, all_roi, first_time, all_numeric_x, cols = \
            Format.format_for_training(train_df_dic, time_mode, train_interval, crypto, \
                                       file_name, features, first_time, all_x, all_y, \
                                           all_roi, look_aheads, markov_number, all_numeric_x, window, tern, price_comp)
    
    # reshapes the targets to 2d arrays 
    all_y = np.reshape(all_y, (-1))
    all_roi = np.reshape(all_roi, (-1))
    return all_x, all_y, all_roi, all_numeric_x, cols

def save_models(to_save, features, crypto, train_interval, thyme, name):
    ''' saves select models and the features they used in the config folder'''
    file = open(config.get_file_paths() + crypto + train_interval + \
                name.replace(' ', '') + thyme.replace('/', '_') + '.pkl', 'wb')
    pickle.dump(to_save, file)
    file.close()
    file = open(config.get_file_paths() + crypto + train_interval + \
                name.replace(' ', '') + thyme.replace('/', '_') + '_FEAT.pkl', 'wb')
    pickle.dump(features, file)
    file.close()

def create_csv(name, df):
    df.to_csv(config.get_file_paths() + name) 
    return

def save_a_priori(file_name, state_dic, features):
    file = open(config.get_file_paths() + file_name, 'wb')
    pickle.dump(state_dic, file)
    file.close()
    file = open(config.get_file_paths() + file_name.replace('.pkl', '_FEAT.pkl'), 'wb')
    pickle.dump(features, file)
    file.close()

def load_multi_training_data(start_cb_num, end_cb_num, time_mode, train_intervals, \
                                 crypto, features, look_aheads):
    file_names = []
    all_x = None
    all_y = None
    all_roi = None
    all_numeric_x = None
    first_time = True
    
    for num in range(start_cb_num, end_cb_num + 1):
        file_names.append('altCB' + str(num) + '.pkl')
    # for each file gets the oscillators, moving averages and start/ end prices
    for file_name in file_names:
        file = open(config.get_file_paths() + file_name, 'rb')
        train_df_dic = pickle.load(file)
        file.close()
        all_x, all_y, all_roi, first_time, all_numeric_x, cols = \
            Format.format_for_multi_training(train_df_dic, time_mode, train_intervals, \
                                             crypto, file_name, features, first_time, \
                                                 all_x, all_y, all_roi, look_aheads, \
                                                     all_numeric_x)
    # reshapes the targets to 2d arrays 
    all_y = np.reshape(all_y, (-1))
    all_roi = np.reshape(all_roi, (-1))
    return all_x, all_y, all_roi, all_numeric_x, cols     
        
        
        
        
    