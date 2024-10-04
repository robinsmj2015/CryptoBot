#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 20:26:13 2021

@author: robinson
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import bootstrap_plot
def print_stuff(tracker, recommender, usd_bal, crypto_bal, trade_limited):
      
    print('''\033[1;37m ============= {0}: from {1} to {2} ============
      
    \033[1;33m Current Price: ${3}, Last Price: ${4}, Starting Price: ${5} 
          
    \033[1;37m Money: ${6:.2f}, USD_bal: ${7:.2f}, Crypto_bal: {8:.4f}, Starting amount: ${9:.2f}
    
     Feeless ROI: {10:.4f}, \033[1;32m (+)ROI Sum: {11:.4f}, \033[1;31m (-)ROI Sum: {12:.4f} 
    
    \033[1;37m Feeless Holding ROI: {13:.4f}, Holding amount: ${14:.2f} 
    
    \033[1;31m Fees Acquired: {15} (@ {16:.2%}), Total fees: ${17:.2f}, \033[1;37m Trade Limited?: {18} 
                  
    \033[1;32m Trading Gains: ${19:.2f}, \033[1;31m Trading Losses: ${20:.2f}, \033[1;37m Trading Sum: ${21:.2f}
    
     Prediction Intervals: {22}, Predictions: {23}, Determined Prediction: {24:.4f}
                  
     \033[1;33m Threshold: {25}, \033[1;37m Threshold Moving Amount: {26}, Threshold Cap: {27} 
     
     Weights: {28}, Policy: {29}
    
    '''.format(tracker.name, tracker.start_time, tracker.scrape_time, \
        tracker.price_list[-1], tracker.price_list[-2], tracker.price_list[0], \
            tracker.money, usd_bal, crypto_bal, tracker.starting_amount, \
                tracker.roi_sum, tracker.pos_roi, tracker.neg_roi, \
                    tracker.holding_roi_sum, tracker.hodl_money_list[-1], \
                        tracker.fee_count, tracker.fee_rate, tracker.fee_total, \
                            trade_limited, tracker.gains, tracker.losses, \
                                tracker.gains - tracker.losses, \
                                    recommender.pred_intervals, \
                                        recommender.predictions, recommender.avg, \
                                            recommender.thresholds, recommender.thresholds_slide_amt, \
                                                recommender.thresholds_cap, recommender.pred_weights, \
                                                    recommender.policy))

    if tracker.graphic_counter == tracker.graphic_interval:
        show_graphs(tracker, recommender)
        tracker.graphic_counter = 0
    else:
        tracker.graphic_counter += 1
    return tracker.graphic_counter


def show_graphs(tracker, recommender):
    fig, (ax00, ax10, ax20) = plt.subplots(nrows = 3, ncols = 1, sharex = True)
    ax00.set_xlabel('trade intervals')
    ax00.set_ylabel('Money (including fees)')
    ax00.set_title('Money (with fees) vs time for ' + tracker.name)
    ax00.plot(tracker.hodl_money_list, color = 'blue', label = 'Holding')
    ax00.plot(tracker.money_list, color = 'yellow', label = 'Your bot')
    ax00.legend(prop = {'size': 10})
    
    ax10.set_ylabel('Feeless ROI')
    ax10.set_title('Feeless ROI vs time for ' + tracker.name)
    ax10.plot(tracker.holding_roi_sum_list, color = 'navy', label = 'holding sum')
    ax10.plot(tracker.roi_sum_list, color = 'gold', label = 'your bot sum')
    ax10.plot(tracker.roi_list, color = 'yellow', label = 'bot moving roi')
    ax10.plot(tracker.holding_roi_list, color = 'cyan', label = 'holding moving roi')
    ax10.legend(prop = {'size': 10})
    
    ax20.set_ylabel('Price')
    ax20.set_title('Price vs time for ' + tracker.name)
    ax20.plot(tracker.price_list, color = 'green', label = 'price')
    ax20.legend(prop = {'size': 10})

def print_training_metrics(evals, key):
    '''display results'''
    print(''' ------------- {0} --------------- \nall_test_accuracies: {1}'''.\
          format(key, np.round(evals['test_accuracy'], 3)))
    for eval_key in evals.keys():
        if not ('time' in eval_key or 'estimator' in eval_key):
            print('''{0}: {1:.3f}'''.format(eval_key, np.mean(evals[eval_key])))

def print_training_rois(key, rois, avg_roi, holding_rois, avg_holding_roi):
    print(''' 
        ----------------- {0} --------------
        Model each fold: {1}
        Avg fold roi: {2}
        Holding each fold: {3}
        Avg holding fold roi: {4} '''.format(key, rois, avg_roi, holding_rois, \
            avg_holding_roi))
                                      
def make_roi_count_bars(counts, averages, features, crypto, interval):
    x_ax_values = np.arange(len(list(counts.keys())))
    fig, (ax0, ax1) = plt.subplots(nrows = 2, ncols = 1, sharex=True)
    
    ax0.set_ylabel('Number of Occurrences')
    ax0.set_title(crypto + ' ' + interval + ' ' + str(features))
    graph0 = ax0.bar(x = x_ax_values, tick_label = list(counts.keys()), align = 'edge', color = 'orange', height = list(counts.values()))
    ax0.bar_label(graph0, padding = 3)
    ax0.grid(visible = True)
    
    ax1.set_xlabel('Feature Values')
    ax1.set_ylabel('Avg ROI')
    ax1.grid(visible = True)
    graph1 = ax1.bar(x = x_ax_values, tick_label = list(averages.keys()), align = 'edge', color = 'purple', height = list(averages.values()))                               
    ax1.bar_label(graph1, padding = 8)
    
    plt.xticks(rotation = 90)
    
def visualize_text_for_pie(pct, data):
    percent = int(np.round(pct/100.*np.sum(data)))
    return "{:.1f}%\n({:d})".format(pct, percent)

def make_transition_pies(state_dic, features, crypto, interval):
    state_dic_keys = list(state_dic.keys())
    num_ax = len(state_dic_keys)
    if num_ax % 5 == 0:
        num_rows = int(num_ax / 5)
    else:
        num_rows = int((num_ax // 5) + 1)
    
    fig, all_axes = plt.subplots(nrows = num_rows, ncols = 5)
    all_axes = all_axes.flatten()
    total_percent = 0
    for idx in range(len(state_dic)):
       state = state_dic[state_dic_keys[idx]]
       total_percent += state.percent_occurrence
       data = list(state.transitions.values())
       wedges, texts, autotexts =  all_axes[idx].pie(data, autopct = lambda pct: visualize_text_for_pie(pct, data))
       all_axes[idx].legend(wedges, list(state.transitions.keys()), loc = "lower left", fontsize = 4, bbox_to_anchor = (-.5, .5))
       plt.setp(autotexts, size = 5, weight = "bold")
       all_axes[idx].set_title('<ROI: {0:.4f} {1} ({2:.2%})>'.format(state.roi, state.count, state.percent_occurrence), fontsize = 9)
    for idx in range(len(state_dic), len(all_axes), 1):
        fig.delaxes(all_axes[idx])
    fig.suptitle('{0} {1} Transitions from {2:.2%} of states using {3}'.format(crypto, interval, total_percent, features))

def make_state_hist(state_dic, features, crypto, interval):
    state_dic_keys = list(state_dic.keys())
    num_ax = len(state_dic_keys)
    if num_ax % 5 == 0:
        num_rows = int(num_ax / 5)
    else:
        num_rows = int((num_ax // 5) + 1)
    fig, all_axes = plt.subplots(nrows = num_rows, ncols=5, sharex=False)
    all_axes = all_axes.flatten() 
    for idx in range(len(state_dic)):
        state = state_dic[state_dic_keys[idx]]
        all_axes[idx].hist(state.rois, color = 'teal', bins = 'auto', density = False, label = ''''{0} \n({1:.4f}, {2:.4f}, {3}) <{4:.4f}, {5:.4f}>'''.format(state.name, state.mean, state.std, state.num, state.q25, state.q50))
        all_axes[idx].set_xlabel('ROIs', fontsize = 8)
        all_axes[idx].set_ylabel('Count', fontsize = 8)
        all_axes[idx].vlines(state.mean, ymin = 0, ymax = 2, color = 'pink', linestyle='dashed')
        
        all_axes[idx].legend(fontsize = 7)
    for idx in range(len(state_dic), len(all_axes), 1):
        fig.delaxes(all_axes[idx])
        fig.suptitle('{0} {1} {2}'.format(crypto, interval, features))

def make_bootstrap_plots(state_dic):
    for state in state_dic.values():
        fig = bootstrap_plot(pd.Series(state.rois))
        fig.suptitle('{0} ~ {1:.4f} ({2:.4f}, {3})'.format(state.name, state.mean, state.std, state.count))
        