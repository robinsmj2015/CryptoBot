#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:52:22 2021

@author: robinson
"""
import Math_Utils
import Format
def predict(x, policy, model, is_invested, depth, action_boost, bayes_cutoff_perc, fee_rate):
    est_roi = 0
    if 'PROB' in policy.upper():
        prediction = model.predict_proba(x)[0]
        try:
            prediction = round(prediction[1], 4) #assigns probability of positive class (price up prediction)
        except TypeError:
            raise AssertionError("If probability mode is True, a probability model must be used")
            
    elif 'BOOL' in policy.upper():
        prediction = model.predict(x)[0]
    
    else:
        state_str = Format.format_for_rl_pred(x[0])
        if state_str not in model:
            print('WARNING no data on state: ' + state_str + '\n Recommending to stay')
            prediction = .5 # neutral
        else:
            est_roi = Math_Utils.calc_for_rl(model, state_str, depth, bayes_cutoff_perc)
        
            if is_invested:
                if est_roi > 0:
                    prediction = 1 #in
                elif est_roi - action_boost + fee_rate < 0:
                    prediction = 0 # out
                else:
                    prediction = .5 # neutral
            else:
                if est_roi < 0:
                    prediction = 0 #out
                elif est_roi + action_boost - fee_rate > 0:
                    prediction = 1 # in
                else:
                    prediction = .5 # neutral
    
    
    return prediction, est_roi



        
            