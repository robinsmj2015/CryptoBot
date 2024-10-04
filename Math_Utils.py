#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:11:15 2021

@author: robinson
"""

def calc_roi(old_price, current_price):
    roi = (current_price - old_price) / old_price
    return roi


def calc_money(crypto_bal, usd_bal, current_price):
    money = usd_bal + crypto_bal * current_price 
    return money

def slide_thresholds(starting_thresholds, thresholds, thresholds_slide_amt, thresholds_cap, money, starting_amt):
   factor = ((money - starting_amt) / starting_amt) / .01
   temp_list = [0, 0]
   if factor > 0:
       temp_list[0] = round(starting_thresholds[0] + factor * thresholds_slide_amt[0], 2) # inc buying threshold
       temp_list[1] = round(starting_thresholds[1] - factor * thresholds_slide_amt[1], 2) # dec selling threshold
       if temp_list[0] <= thresholds_cap[0]:
           thresholds[0] = temp_list[0]
       if temp_list[1] >= thresholds_cap[1]:
           thresholds[1] = temp_list[1]
   
   else:
       thresholds = starting_thresholds
   print(starting_thresholds, thresholds, factor, starting_amt, money)
   return thresholds

def calc_for_rl(state_dic, state_str, depth, bayes_cutoff_perc):
   
   # assume undiscounted rewards but only going down certain depth
   # if sample size is low mean might not reflect actual mean
   state = state_dic[state_str]
   count = state.count
   parent_reward = state.roi
   child_reward = 0
   depth = 2 # FOR NOW
   # add in boost
   for key in state.transitions.keys():
       if (state.transitions[key] / count) > (bayes_cutoff_perc / 100):
           if key in state_dic: # just in case state was only reached at end of data sample 
               child_reward += state_dic[key].roi * (state.transitions[key] / count)
   return round(parent_reward + child_reward, 5)
   
    