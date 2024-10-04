#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:43:43 2021

@author: robinson
"""
from datetime import datetime
import time

# import Train
import Trader
import Format
import Controller
import File_Utils
import Math_Utils
import Visualize
import Scrape_Utils

from Explore_Data import State

# makes a trader instance # all were trained for 2 hrs!!!!
trade_interval = '2h'
crypto = 'ADA'
exchange = 'COINBASE'
policy = 'BOOL_VOTE' # BOOL_VOTE, BOOL_JURY, PROB_VOTE, PROB_JURY, RL_VOTE
reduce_trading = False
starting_amount = 1000

model_files = ['ADA1mSupportVectorMachine02_21_22_11_56.pkl', 'ADA5mSupportVectorMachine02_21_22_11_55.pkl', \
               'ADA15mSupportVectorMachine02_21_22_11_55.pkl', 'ADA30mSupportVectorMachine02_21_22_11_55.pkl', \
                   'ADA1hSupportVectorMachine02_21_22_11_54.pkl', 'ADA2hSupportVectorMachine02_21_22_11_53.pkl']

pred_intervals = ['1m', '5m', '15m', '30m', '1h', '2h']
name = 'ada_2h_bot1'
graphic_interval = 2
pred_weights = [] # sum to 1
thresholds = [.52, .52] # used in prob vote
thresholds_slide_amt = [0, 0] # for every 1% increase in money above starting amount, buying will be tougher and selling easier by this amount
thresholds_cap = [.52, .52] # most extreme values allowed
fee_rate = 0.001

action_boost = [0.0009] * 6 # boosts the q value for taking an action (rather than staying same)
depths = [2] * 6 # depth search level for transition tree (only applicable if policy is RL)
# ADD IN CONFIDENCE VALUES FOR REWARDS AND TRANSITIONS
bayes_cutoff_perc = [3] * 6
ada_2h_bot1 = Trader.Trader(trade_interval, \
                            crypto, policy, reduce_trading, starting_amount, \
                                model_files, pred_intervals, graphic_interval, name, \
                                    exchange, pred_weights, thresholds, \
                                        fee_rate, thresholds_slide_amt, thresholds_cap, \
                                            depths, action_boost, bayes_cutoff_perc)
  
trading_bots = [ada_2h_bot1]



print('''Hi there! Im Snottie-Bot - your personal trading bot! 
      We're about to start trading, so wish me luck (since it's your money)! \033[5;35m
             __
     _(\    |@@|
    (__/\__ \--/ __
       \___|----|  |   __
           \ }{ /\ )_ / _\
           /\__/\ \__O (__
          (--/\--)    \__/
          _)(  )(_
         `---''---`) \033[1;37m \n''')



# makes sure we are at the top of the minute for scraping and trading
sec_now = '3'
while int(sec_now) >= 3:
    sec_now = datetime.now().strftime('%S')
    time.sleep(1)
num_scrapes_in_min = 0
df_dic = {}
alt_df_dic = {}
last_min = ''



while True:
    time.sleep(1)
    
    for bot in trading_bots:
        # the first time (bot) of the loop we will scrape all cryptos for the appropriate intervals (1m, 1m and 5m, ...etc)
        if bot is trading_bots[0]:
            # scrape if time conditions are met (sec ~ 6 or sec ~ 36)
            last_min, num_scrapes_in_min, df_dic, mins, hrs , trading_bots, bot.tracker.scrape_time, alt_df_dic = \
                Scrape_Utils.check_for_scrape(last_min, num_scrapes_in_min, df_dic, trading_bots, exchange, alt_df_dic)
        else:
            bot.tracker.scrape_time = trading_bots[0].tracker.scrape_time
        # sees if we have scraped twice this min and the bot has not yet made a prediction
        if num_scrapes_in_min == 2 and not bot.made_prediction:
            # sees if the bot trades minutely, 5 mins, 15 min etc and then decides to make a prediction if so
            if ('m' in bot.trade_interval and int(mins) in bot.trade_times) or \
                ('h' in bot.trade_interval and int(hrs) in bot.trade_times and int(mins) == 0):
                    # formatting for prediction - using scraped data above
                    bot.x, bot.current_price = Format.format4predicting(alt_df_dic, bot.crypto, \
                                                        bot.trade_interval, bot.recommender.pred_intervals, \
                                                            bot.model_features)
                    
                    # recommendation made
                    bot.recommender.make_prediction_list(bot.x, bot.models)
                    bot.made_prediction = True
                    # updates total money (crypto * price + USD)
                    bot.tracker.money = Math_Utils.calc_money(bot.crypto_bal, bot.usd_bal, bot.current_price) 
                    if bot.first_time:
                        # for the bots first time - updates price list, graphic counter and records start time
                        bot.tracker.start_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                        bot.tracker.price_list.append(bot.current_price)
                        bot.tracker.graphic_counter += 1
                        bot.first_time = False
                    else:
                        # calculates roi if not first time and determines if this holding roi is also the bot's roi (if bot is invested)
                        bot.tracker.holding_roi = Math_Utils.calc_roi(bot.old_price, bot.current_price)
                        if bot.recommender.is_invested: 
                            bot.tracker.roi = bot.tracker.holding_roi
                        else:
                            bot.tracker.roi = 0
                        # updates trackers and displays visuals
                        bot.tracker.update_trackables(bot.current_price, bot.usd_bal, bot.crypto_bal)
                        bot.tracker.graphic_counter = Visualize.print_stuff(bot.tracker, \
                                                                            bot.recommender, \
                                                                                bot.usd_bal, bot.crypto_bal, \
                                                                                    bot.trade_limited)  
                    # decides to buy/ sell as appropriate (looks at trade_limited if reduce_trading enabled)   
                    # WARNING - currently reduce_trading does not look at price changes
                    if bot.recommender.recommend_in and not bot.recommender.is_invested and not bot.trade_limited:
                        bot.buy_all()
                        bot.recommender.is_invested = True
                        if bot.reduce_trading:
                            bot.trade_limited = True
                    elif not bot.recommender.recommend_in and bot.recommender.is_invested and not bot.trade_limited:
                        bot.sell_all() 
                        bot.recommender.is_invested = False
                        if bot.reduce_trading:
                            bot.trade_limited = True
                    elif bot.trade_limited:
                        bot.trade_limited = False
                    bot.old_price = bot.current_price
                    if 'PROB' in bot.recommender.policy and bot.recommender.thresholds_slide_amt != [0, 0]:
                        bot.recommender.thresholds = \
                            Math_Utils.slide_thresholds(bot.recommender.starting_thresholds, \
                                                        bot.recommender.thresholds, \
                                                            bot.recommender.thresholds_slide_amt, \
                                                                bot.recommender.thresholds_cap, \
                                                                    bot.tracker.money, \
                                                                        bot.starting_amount)
# ideas:
    # multiple reference points
    # multiple intervals ahead - look aheads
    # trading view data is not markov - use prior states too (1-2 more)
    
    # maybe the question is not what the price will be at when, but when will the price be at

# regression!!!
# use numeric features!

# downsampling and upweighting!
# not randomizing the data when training!
# multiple models based on different file nums - same means? bootstrap plot



# CNN classifier
# with GAN?
# clean data - remove 0? rn the classifier is up or not up (ie same falls under down)


''' https://arxiv.org/pdf/1207.1386.pdf,
http://ee266.stanford.edu/lectures/inf_horiz.pdf,
http://people.ee.duke.edu/~lcarin/Xuejun4.30.2010.pdf'''

