#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:02:52 2021

@author: robinson
"""
import Math_Utils
import File_Utils
import Recommender
import Tracker

class Trader():
    def __init__(self, trade_interval, crypto, policy, reduce_trading, starting_amount, \
                 model_files, pred_intervals, graphic_interval, name, exchange='BINANCE', pred_weights=None, \
                     thresholds=[.5, .5], fee_rate=0.001, thresholds_slide_amt=[0, 0], thresholds_cap=[.7, .3], depths=None, action_boost=None, bayes_cutoff_perc=None):
        
        self.usd_bal = starting_amount
        self.crypto_bal = 0
        self.current_price = 0
        self.old_price = 0
        
        self.crypto = crypto
        self.fee_rate = fee_rate
        self.starting_amount = starting_amount
        
        self.trade_interval = trade_interval
        self.exchange = exchange
        
       
        self.reduce_trading = reduce_trading
        self.trade_limited = False
        self.model_files = model_files
        
        self.first_time = True
        self.made_prediction = False
       
        
        feature_files = []
        for model_file in model_files:
            feature_files.append(model_file.replace('.pkl', '_FEAT.pkl'))
        self.models = File_Utils.load_models(model_files)
        self.model_features = File_Utils.load_features(feature_files)
        self.recommender = Recommender.Recommender(policy, crypto, fee_rate, pred_intervals, pred_weights, thresholds, thresholds_slide_amt, thresholds_cap, depths, action_boost, bayes_cutoff_perc)
        self.tracker = Tracker.Tracker(starting_amount, graphic_interval, name, fee_rate)
        
        self.get_trade_times()

    def get_trade_times(self):
        step = int(self.trade_interval[0:-1])
        if self.trade_interval.endswith('m'):
            self.trade_times = [i for i in range(0, 60, step)]
        else:
            self.trade_times = [i for i in range(0, 24, step)]
        

    def buy(self, crypto_amount):
        old_crypto_amount = crypto_amount
        usd_to_subtract = crypto_amount * self.current_price
        usd_to_subtract += usd_to_subtract * (1-self.fee_rate)
        
        if self.usd_bal < usd_to_subtract:
            usd_to_subtract = self.usd_bal
            crypto_amount = usd_to_subtract/self.current_price
            print('Insufficient funds to buy ' + str(old_crypto_amount) + self.crypto + ' at ' + str(self.current_price) + ' buying ' + str(crypto_amount) + ' instead.')

        self.crypto_bal += crypto_amount
        self.usd_bal -= usd_to_subtract
        print('Buying ' + str(old_crypto_amount) + self.crypto + ' at ' + str(self.current_price) + '.')
        return crypto_amount
           
    def sell(self, crypto_amount):
        
        old_crypto_amount = crypto_amount
        if self.crypto_bal < crypto_amount:
            crypto_amount = self.crypto_bal
            print('Insufficient funds to sell ' + str(old_crypto_amount) + self.crypyto + ' at ' + str(self.self.current_price) + ' selling ' + str(crypto_amount) + ' instead.')
        usd_to_add = crypto_amount * self.current_price
        usd_to_add -= usd_to_add * (1-self.fee_rate)
        self.crypto_bal -= crypto_amount
        self.usd_bal += usd_to_add
        print('Selling ' + str(old_crypto_amount) + ' at ' + self.crypto + str(self.current_price) + '.')
        return crypto_amount
    
    def buy_all(self):
        # changed - to * below ???
        self.tracker.fee_total += self.usd_bal * self.fee_rate 
        crypto_amount = (self.usd_bal * (1 - self.fee_rate)) / self.current_price

        self.crypto_bal = crypto_amount
        self.usd_bal = 0
        
        self.tracker.fee_count += 1
        
        self.tracker.fee_sum_list.append(self.tracker.fee_total)
        
        print('\033[1;32m {0} is buying {1:.4f} {2} at ${3} \033[1;37m'.format(self.tracker.name, crypto_amount, self.crypto, self.current_price))
    
    def sell_all(self):
        
        self.tracker.fee_total += self.current_price * self.crypto_bal * self.fee_rate
        amount_usd = self.current_price * self.crypto_bal
        crypto_amount = self.crypto_bal
        self.crypto_bal = 0
        self.usd_bal = amount_usd * (1 - self.fee_rate)
        
        self.tracker.fee_count += 1
        
        self.tracker.fee_sum_list.append(self.tracker.fee_total)
        
        print('\033[1;31m {0} is selling {1:.4f} {2} at ${3} \033[1;37m'.format(self.tracker.name, crypto_amount, self.crypto, self.current_price))

    