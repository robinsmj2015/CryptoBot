#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 10:43:02 2021

@author: robinson
"""

''' tracks trading based data like fees, rois, prices, etc - one tracker instance per trader '''
class Tracker:
    def __init__(self, starting_amount, graphic_interval, name, fee_rate):
        self.money_list = []
        self.roi = 0 # updated in math utils
        self.roi_list = []
        self.roi_sum = 0
        self.roi_sum_list = []
        self.holding_roi = 0
        self.holding_roi_sum = 0
        self.holding_roi_list = []
        self.holding_roi_sum_list = []
       
        self.price_list = []
       
        self.pos_roi = 0
        self.neg_roi = 0
        self.fee_rate = fee_rate
        self.starting_amount = starting_amount
        self.money = starting_amount # updated in math utils
        
        
        self.hodl_money_list = [round(starting_amount * (1 - fee_rate), 2)]
       
        
        self.gains = 0
        self.losses = 0
        self.crypto_bal_list = []
        self.usd_bal_list = []
        
        self.fee_count = 0 # updated in trader buy all /sell all functions
        self.fee_total = 0
        self.fee_sum_list = []
        
        
        self.trip_wire = False
        self.graphic_interval = graphic_interval
        self.graphic_counter = 0
        
        self.name = name
        
        
    ''' updates lists of data - mainly to be used in graphing data '''
    def update_trackables(self, current_price, usd_bal, crypto_bal):
        self.price_list.append(current_price)
        self.money_list.append(self.money)
        self.roi_list.append(self.roi)
        self.roi_sum += self.roi
        self.roi_sum_list.append(self.roi_sum)
        self.holding_roi_list.append(self.holding_roi)
        self.holding_roi_sum += self.holding_roi
        self.holding_roi_sum_list.append(self.holding_roi_sum)
        self.hodl_money_list.append((1 + self.holding_roi) * self.hodl_money_list[-1])
        self.usd_bal_list.append(usd_bal)
        self.crypto_bal_list.append(crypto_bal)
        if self.roi > 0:
            self.pos_roi += self.roi
            self.gains += crypto_bal * self.roi * current_price
        else:
            self.neg_roi += -1 * self.roi
            self.losses += crypto_bal * self.roi * current_price * -1
        