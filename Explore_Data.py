#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:29:17 2022

@author: robinson
"""
import pandas as pd
import File_Utils
import Visualize
import config
import Format
from pandas.plotting import lag_plot, autocorrelation_plot, bootstrap_plot


class State():
    def __init__(self, name, total):
        self.name = name
        self.transitions = {}
        self.count = 1
        self.rois = []
        
def run():
    bootstrap_plots = False
    just_make_csv = False       
    save = True
    bool_roi = True
    numerical = False
    time_mode = '1st'
    interval = '30m'
    crypto = 'ADA'
    file_name = crypto + interval + '.csv'  
    features = ['EMA10', 'SMA10', 'EMA20', 'SMA20', 'EMA50', 'SMA50', 'EMA100', 'SMA100', \
     'EMA200', 'SMA200', 'UO', 'AO', 'ADX', 'RSI', 'Mom', 'HullMA', 'STOCH.K', 'VWMA', 'Stoch.RSI', \
         'BBP', 'MACD', 'Ichimoku', 'CCI', 'W%R', 'EMA30', 'SMA30']
    
    features = ['EMA20', 'EMA50', 'SMA50', 'SMA200', 'EMA30', 'HullMA', 'VWMA', 'Mom']
    
    file_num = 36
    cb_num = 24
    look_aheads = 1
    markov_number = 1
    remove_threshold = 50 # removes state from graph is not reached more than this many times
    pie_threshold = 10 #percent
    all_x, all_y, all_roi, all_numeric_x, cols = File_Utils.load_training_data(time_mode, \
                                                          interval, crypto, features, \
                                                            file_num, look_aheads, markov_number,cb_num)
    
      
    '''
    t = pd.Series(all_roi)
    lag_plot(t) 
    autocorrelation_plot(t)
    '''
    if just_make_csv:
        if bool_roi:
            the_y = all_y
        else:
            the_y = all_roi
        if numerical:
            features = [(lambda col: col[0])(col) for col in cols]
            df = Format.format_for_csv(the_y, all_numeric_x, features, bool_roi)
        else:
            df = Format.format_for_csv(the_y, all_x, features, bool_roi)
        File_Utils.create_csv(file_name, df) 
        print('{0} was created in {1}. \nTo run rest of program change just_make_csv to False.'.format(file_name, config.get_file_paths()))
        raise SystemExit()
        
    
    old_feat_str = ''
    count_dic = {}
    roi_dic = {}
    state_dic = {}
    total = all_x.shape[0] - 1
    for sample_num in range(total):
        # makes a str corresponding to the sample id (ie the value of each feature for the sample)
        new_feat_str = ''
        for feature_num in range(all_x.shape[1]):
            new_feat_str += str(all_x[sample_num][feature_num])
        # makes a unique State instance for each unique sample, also counts times reached state in a dic
        if new_feat_str not in count_dic.keys():
            count_dic[new_feat_str] = 1
            roi_dic[new_feat_str] = round(all_roi[sample_num], 4)
            state_dic[new_feat_str] = State(new_feat_str, total)
            state_dic[new_feat_str].roi = round(all_roi[sample_num], 4)
        # for samples that have been reached, adds 1 to count and updates avg roi and percent occurred    
        else:
            count_dic[new_feat_str] += 1
            state_dic[new_feat_str].count += 1 
            
            roi_dic[new_feat_str] = round(roi_dic[new_feat_str] + \
               (1 / count_dic[new_feat_str]) * \
                   (all_roi[sample_num] - roi_dic[new_feat_str]), 4)
            state_dic[new_feat_str].roi = round(state_dic[new_feat_str].roi + \
               (1 / state_dic[new_feat_str].count) * \
                   (all_roi[sample_num] - state_dic[new_feat_str].roi), 4)    
        state_dic[new_feat_str].rois.append(all_roi[sample_num])
        state_dic[new_feat_str].percent_occurrence = round(state_dic[new_feat_str].count / total, 4)
        state_dic[new_feat_str].description = state_dic[new_feat_str].name + ' $ ' + str(state_dic[new_feat_str].roi)
        # tracks what state comes after the sample (ie transitions)
        if sample_num != 0: 
            if new_feat_str not in state_dic[old_feat_str].transitions.keys():
                state_dic[old_feat_str].transitions[new_feat_str] = 1 
           
            else: 
                state_dic[old_feat_str].transitions[new_feat_str] += 1 
            
        old_feat_str = new_feat_str
    
    if save:
        File_Utils.save_a_priori(file_name.replace('.csv', '.pkl'), state_dic, features)
    # removes states that didnt occur very much
    state_keys = list(state_dic.keys())
    for state_key in state_keys:
       if state_dic[state_key].count <= remove_threshold:
           state_dic.pop(state_key)
       
           
    
    for state in state_dic.values():   
        series = pd.Series(state.rois)
        state.std = series.std()
        state.q25 = series.quantile(.25)
        state.q50 = series.quantile(.50)
        state.q75 = series.quantile(.75)
        state.mean = series.mean()
        state.max = series.max()
        state.min = series.min()
        state.num = series.count()
        trans_keys = list(state.transitions.keys())
        other_transition_sum = 0
        # removes transitions from state if they didnt occur much and lumps them together in 'other states'
        for trans_key in trans_keys:
            percent = round(100 * state.transitions[trans_key] / state.count, 3)
            if percent <= pie_threshold:
                other_transition_sum += state.transitions[trans_key]
                state.transitions.pop(trans_key)
                
        state.transitions['Other'] = round(other_transition_sum, 3)
        # renames the transition to include 'self' if the state transitions to itself
        if state.name in state.transitions.keys():
            self_transition = state.transitions[state.name]
            state.transitions.pop(state.name)
            state.transitions['Self (' + state.name + ')'] = self_transition
    
    # for first fig showing counts and rois removes states that didnt occur much
    keys = list(count_dic.keys())
    for key in keys:
        if count_dic[key] <= remove_threshold:
            count_dic.pop(key)
            roi_dic.pop(key)
    
    
    # be aware some states are popped
    # plot each state as a hist (roi)
    # save as df - get stats (SD, mean)
    # also make bins and compare SD / means



    
    # generates figures
    Visualize.make_roi_count_bars(count_dic, roi_dic, features, crypto, interval)
    Visualize.make_transition_pies(state_dic, features, crypto, interval)
    Visualize.make_state_hist(state_dic, features, crypto, interval)
    if bootstrap_plots:
        Visualize.make_bootstrap_plots(state_dic)
    #******************** not perfect due to interuptions between files
    
if __name__ == '__main__':
    run()  
    
    
    