#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:43:52 2021

@author: robinson
"""
import Predict
import numpy as np
# maybe make just a file
# recommend_in is the recommendation an instance outputs after calling get_all_predictions
class Recommender:
    def __init__(self, policy, crypto, fee_rate, pred_intervals, pred_weights, thresholds, thresholds_slide_amt, thresholds_cap, depths, action_boost, bayes_cutoff_perc):
        self.policy = policy
        self.crypto = crypto
        self.thresholds = thresholds
        self.starting_thresholds = thresholds.copy()
        self.pred_weights = pred_weights
        self.is_invested = False
        self.pred_intervals = pred_intervals
        self.avg = 'N/A'
        self.thresholds_slide_amt = thresholds_slide_amt
        self.thresholds_cap = thresholds_cap
        
        self.fee_rate = fee_rate
        self.depths = depths
        self.action_boost = action_boost
        self.bayes_cutoff_perc = bayes_cutoff_perc

    def make_prediction_list(self, x, models_to_use):
        self.predictions = []
        self.est_rois = []
        
        model_num = 0
        for model in models_to_use:
            
            prediction, est_roi = Predict.predict(x[model_num], self.policy, model, self.is_invested, self.depths[model_num], self.action_boost[model_num], self.bayes_cutoff_perc[model_num], self.fee_rate)
            model_num += 1
            self.est_rois.append(est_roi)
            self.predictions.append(prediction)
        
        self.determine_policy()

    def determine_policy(self):
        if 'PROB' in self.policy:
            if 'VOTE' in self.policy:
                self.prob_vote()
            else:
                self.prob_jury()
        elif 'BOOL' in self.policy:
            if 'VOTE' in self.policy:
                self.bool_vote()
            else:
                self.bool_jury()
        if 'RL' in self.policy:
            if 'VOTE' in self.policy:
                self.rl_vote()
            
    
    def prob_vote(self):
        if type(self.pred_weights) is None:
            self.pred_weights = [round(1 / len(self.pred_intervals), 3)] * len(self.pred_intervals)
        self.avg = 0
        for idx in range(len(self.predictions)):
            self.avg += self.predictions[idx] * self.pred_weights[idx]
        
        if self.is_invested:
            # if invested currently
            if 1 - self.avg > self.thresholds[1]:
                # if chance of negative class exceeds sell threshold
                self.recommend_in = False
            else:
                self.recommend_in = True
        else:
            # if not invested currently
            if self.avg > self.thresholds[0]:
                 # if chance of postive class exceeds buy threshold
                self.recommend_in = True
            else:
                self.recommend_in = False
            
 
    def prob_jury(self):
        if self.is_invested:
            # if currently invested
            count = 0
            for prediction in self.predictions:
                if 1 - prediction > self.thresholds[1]: 
                    # if chance of negative class exceeds sell threshold
                    count += 1
            if count == len(self.predictions):
                # if all predictions say selling is worth it - recommends selling
                self.recommend_in = False
            else:
                # if not... stay invested
                self.recommend_in = True
        else:
            # if not currently invested
            count = 0
            for prediction in self.predictions:
                if prediction > self.thresholds[0]:
                    # if chance of postive class exceeds buy threshold
                    count += 1
                if count == len(self.predictions):
                    # if all say buy... recommend buying 
                    self.recommend_in = True
                else:
                    # if not... recommends staying out of the market
                    self.recommend_in = False
        
    def bool_vote(self):
        count = 0
        for prediction in self.predictions:
            if prediction:
                count += 1
        if count > (len(self.predictions) - count):
            self.recommend_in = True
        elif count < (len(self.predictions) - count):
            self.recommend_in = False
        else:
            self.recommend_in = self.is_invested
        self.avg = count
            
    def bool_jury(self):
        count = 0
        for prediction in self.predictions:
            if prediction:
                count += 1
        if count == 0:
            self.recommend_in = False
        elif count == len(self.predictions):
            self.recommend_in = True
        else:
            self.recommend_in = self.is_invested
        
       
    def rl_vote(self):
        self.avg = sum(self.predictions) / len(self.predictions)
        if self.avg > .5:
            self.recommend_in = True
        elif self.avg < .5:
            self.recommend_in = False
        else:
            self.recommend_in = self.is_invested
        self.avg = np.mean(self.est_rois)
        
   
    
    
    
    
    
        
   