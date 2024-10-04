#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:57:33 2021

@author: robinson

get data
"""

import time
from tradingview_ta import get_multiple_analysis
import datetime
import os
import pandas as pd
import pickle
import config


class Scraper():
    def __init__(self, intervals, cryptocurrencies, exchange, file_name):
        self.intervals = intervals # intervals to scrape
        self.cryptocurrencies = cryptocurrencies # cryptos to scrape
        self.exchange = exchange # cryptocurrency exchange
        self.file_name = file_name # what to save the file as
        self.api_dic = {} # handlers to scrape data by interval
        self.is_initializing = True # first time through
        self.xchange_n_crypto = [] # list containing exhange and cryptos like COINBASE:ADAUSD
        self.df = {} # dictionary of dataframes used for saving the data
        self.temp_df = {} # dictionary of dataframes 
        self.scraping_counter = 0
        self.here = False
        for crypto in cryptocurrencies:
            self.xchange_n_crypto.append(exchange + ':' + crypto + 'USD')
    
    def save(self):
        '''saves the dictionary to a file'''
        fi = open(config.get_file_paths() + self.file_name, 'wb')
        pickle.dump(self.df, fi)
        fi.close()
    
    def get_data(self):
        '''scrapes'''
        data_dic = {} # dictionary containing the scraped data
        # the time at which data was scraped - to be used as index for dataframe
        thyme = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        self.api_dic[self.interval] = get_multiple_analysis(screener = "crypto", interval = self.interval, \
                                                                symbols = self.xchange_n_crypto)
        for crypto in self.cryptocurrencies:
            
            # stores scraped data in lists for oscillators, moving averages and indicators (keys and values)
            data_dic[crypto + self.interval] = self.api_dic[self.interval][self.exchange + ':' + crypto + 'USD']
            temp_mas_val = list(data_dic[crypto + self.interval].moving_averages['COMPUTE'].values())
            mas_key = list(data_dic[crypto + self.interval].moving_averages['COMPUTE'].keys())
            temp_osc_val = list(data_dic[crypto + self.interval].oscillators['COMPUTE'].values())
            osc_key = list(data_dic[crypto + self.interval].oscillators['COMPUTE'].keys())
            ind_val = list(data_dic[crypto + self.interval].indicators.values())
            ind_key = list(data_dic[crypto + self.interval].indicators.keys())  
            
            # makes each indicator a -1 for sell, 1 for buy and 0 for neutral
            mas_val = []
            osc_val = []
            for element in temp_mas_val:
                if element.upper().startswith('B'):
                    mas_val.append(1)
                elif element.upper().startswith('S'):
                    mas_val.append(-1)
                else:
                    mas_val.append(0)   
            for element in temp_osc_val:
                if element.upper().startswith('B'):
                    osc_val.append(1)
                elif element.upper().startswith('S'):
                    osc_val.append(-1)
                else:
                    osc_val.append(0)
        
            # makes 3 dataframes
            self.temp_df[crypto + self.interval + 'MA'] = pd.DataFrame([mas_val], index = [thyme], \
                                                                       columns = [mas_key])
            self.temp_df[crypto + self.interval + 'OSC'] = pd.DataFrame([osc_val], index = [thyme], \
                                                                        columns = [osc_key])
            self.temp_df[crypto + self.interval + 'IND'] = pd.DataFrame([ind_val], index = [thyme], \
                                                                        columns = [ind_key])
           
        # combines data frames if appropriate
        
        if self.is_initializing:
            self.df = self.temp_df.copy()
        else:
           
            self.combine_df()
                      
    def combine_df(self):
        ''' combines data frames within the self.df dictionary '''
        for crypto in cryptocurrencies:
            self.df[crypto + self.interval + 'MA'] = \
                self.df[crypto + self.interval + 'MA'].append(self.temp_df[crypto + self.interval + 'MA'])
            self.df[crypto + self.interval + 'OSC'] = \
                self.df[crypto + self.interval + 'OSC'].append(self.temp_df[crypto + self.interval + 'OSC'])
            self.df[crypto + self.interval + 'IND'] = \
                self.df[crypto + self.interval + 'IND'].append(self.temp_df[crypto + self.interval + 'IND'])
           
    def mini_scrape(self):
        thyme = datetime.datetime.now().strftime('%H:%M:%S')
        hrs, mins, secs = thyme.split(':')
        
        if self.is_initializing:
            for interval in self.intervals:
                self.interval = interval
                self.get_data()
            self.is_initializing = False
        else:
            self.interval = '1m'
            self.get_data()
            if mins.endswith('00'):
                if int(hrs) % 2 == 0:
                    self.interval = '2h'
                    self.get_data()
                    self.interval = '1h'
                    self.get_data()
                else:
                    self.interval = '1h'
                    self.get_data()
                self.interval = '30m'
                self.get_data()
                self.interval = '15m'
                self.get_data()
            elif mins.endswith('30'):
                self.interval = '30m'
                self.get_data()
                self.interval = '15m'
                self.get_data()
            elif mins.endswith('15') or mins.endswith('45'):
                self.interval = '15m'
                self.get_data()
            if int(mins) % 5 == 0:
                self.interval = '5m'
                self.get_data()
        
        self.save()
    def run(self):
        ''' runs the scraper '''
        # displays start time and graphic
        print(''' \n SCRAPING... started at {} '''.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print('''     
                     
          [_|_/ 
           //
         _//    __
        (_|)   |@@|
         \ \__ \--/ __
          \o__|----|  |   __
              \ }{ /\ )_ / _\
              /\__/\ \__O (__
             (--/\--)    \__/
             _)(  )(_
            `---''---` ''')  
        # loops though each time frame, setting the interval as appropriate, saving the data and sleeping 
        while True:
            thyme = datetime.datetime.now().strftime('%H:%M:%S')
            hrs, mins, secs = thyme.split(':')
            for i in range(2):
                self.interval = '2h'
                self.get_data()
                for j in range(2):
                    self.interval = '1h'
                    self.get_data()
                    for k in range(2):
                        self.interval = '30m'
                        self.get_data()
                        for l in range(2):
                            self.interval = '15m'
                            self.get_data()
                            for m in range(3):
                                self.interval = '5m'
                                self.get_data()
                                for n in range(5):
                                    self.interval = '1m'
                                    self.get_data()
                                    self.save()
                                    self.is_initializing = False
                                    time.sleep(60)
''' ========================  MAIN ============================================= '''                                    
intervals = ['1m', '5m', '15m', '30m', '1h', '2h'] 
cryptocurrencies = ['ETH', 'BTC', 'ADA', 'DOT', 'DASH', 'MANA', 'BAT', 'DOGE', '1INCH', 'UNI']
exchange = 'BINANCE'

if __name__ == '__main__':   
    # determines file name 
    files = os.listdir(config.get_file_paths())
    num = -1 # to be used in file name
    max_num = -1 # to be used in file name (highest number in file path)
    for file in files:
        if file.endswith('.pkl') and (file.startswith('df')):
           idx = file.index('.pkl')
           if file[idx - 1].isnumeric():
               if file[idx - 2].isnumeric():
                   num = int(file[idx - 2: idx]) + 1
               else:
                   num = int(file[idx - 1]) + 1
        if num > max_num:
            max_num = num
    if max_num == -1:
        max_num = ''
    # checks with user if file name is fine and then starts the scrape             
    file_name = input('Is df' + str(max_num) + '.pkl ok as the file name? (hit enter - if not type in the file name): ')               
    if file_name == '':
        file_name = 'BIdf' + str(max_num) + '.pkl'
    
    robot = Scraper(intervals, cryptocurrencies, exchange, file_name)
    robot.run()

# remove 1st entry of df
# through BIdf8.pkl  - 1 hr data is not right