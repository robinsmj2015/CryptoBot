#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 09:47:31 2021

@author: robinson
"""
from tradingview_ta import get_multiple_analysis
from datetime import datetime
                
import Format
import File_Utils

CRYPTOS = ['ETH', 'BTC', 'ADA', 'DOT', 'DASH', 'MANA', 'BAT', 'DOGE', '1INCH', 'UNI']


def scrape(interval, exchange):
        exchange_w_cryptos = []
        for crypto in CRYPTOS:
            exchange_w_cryptos.append(exchange + ':' + crypto + 'USD')
        scrape_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        scraped_dic = get_multiple_analysis("crypto", interval, exchange_w_cryptos)
        return scrape_time, scraped_dic

def check_for_scrape(last_min, num_scrapes_in_min, df_dic, trading_bots, exchange, alt_df_dic):
    time_now = datetime.now().strftime('%H:%M:%S')
    hrs, mins, secs = time_now.split(':')
    scrape_time = ''
    modified_trading_bots = []
    if last_min != mins:
       num_scrapes_in_min = 0 
       modified_trading_bots = []
       for bot in trading_bots:
           bot.made_prediction = False
           modified_trading_bots.append(bot)
    if int(secs) > 6 and num_scrapes_in_min == 0:
        num_scrapes_in_min += 1
        
        intervals = get_intervals_for_scrape()
        for interval in intervals:
            scrape_time, scraped_dic = scrape(interval, exchange)
            df_dic = Format.format_for_saving(scraped_dic, df_dic, scrape_time, interval, exchange, CRYPTOS)
        is_alt_df = False
        File_Utils.create(df_dic, is_alt_df)
        for interval in ['1m', '5m', '15m', '30m', '1h', '2h']:
            scrape_time, scraped_dic = scrape(interval, exchange)
            alt_df_dic = Format.format_for_saving(scraped_dic, alt_df_dic, scrape_time, interval, exchange, CRYPTOS)
         
        is_alt_df = True
        File_Utils.create(alt_df_dic, is_alt_df)
        last_min = mins
        
    elif int(secs) > 36 and num_scrapes_in_min == 1:
        num_scrapes_in_min += 1
        
        intervals = get_intervals_for_scrape()
        for interval in intervals:
            scrape_time, scraped_dic = scrape(interval, exchange)
            df_dic = Format.format_for_saving(scraped_dic, df_dic, scrape_time, interval, exchange, CRYPTOS)
        is_alt_df = False
        File_Utils.create(df_dic, is_alt_df)
        for interval in ['1m', '5m', '15m', '30m', '1h', '2h']:
            scrape_time, scraped_dic = scrape(interval, exchange)
            alt_df_dic = Format.format_for_saving(scraped_dic, alt_df_dic, scrape_time, interval, exchange, CRYPTOS)
         
        is_alt_df = True
        File_Utils.create(alt_df_dic, is_alt_df)
    
    if modified_trading_bots == []:
        modified_trading_bots = trading_bots
    return last_min, num_scrapes_in_min, df_dic, mins, hrs, modified_trading_bots, scrape_time, alt_df_dic


def get_intervals_for_scrape():
    time_now = datetime.now().strftime('%H:%M')
    hrs, mins = time_now.split(':')
    scrape_intervals = ['1m']
    if int(mins) % 5 == 0:
        scrape_intervals.append('5m')
        if int(mins) % 15 == 0:
            scrape_intervals.append('15m')
            if int(mins) % 30 == 0:
                scrape_intervals.append('30m')
                if int(mins) == 0:
                    scrape_intervals.append('1h')
                    if int(hrs) % 2 == 0:
                        scrape_intervals.append('2h')
    return scrape_intervals
    