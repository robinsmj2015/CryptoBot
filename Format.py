#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 12:02:30 2021

@author: robinson
"""
import numpy as np
import pandas as pd



def format_for_saving(scraped_dic, df_dic, scrape_time, interval, EXCHANGE, CRYPTOS):
    rekeyed_data_dic = {}
    temp_df_dic = {}
    for crypto in CRYPTOS:
     # stores scraped data in lists for oscillators, moving averages and indicators (keys and values)
        rekeyed_data_dic[crypto] = scraped_dic[EXCHANGE + ':' + crypto + 'USD']
        temp_mas_val = list(rekeyed_data_dic[crypto].moving_averages['COMPUTE'].values())
        mas_key = list(rekeyed_data_dic[crypto].moving_averages['COMPUTE'].keys())
        temp_osc_val = list(rekeyed_data_dic[crypto].oscillators['COMPUTE'].values())
        osc_key = list(rekeyed_data_dic[crypto].oscillators['COMPUTE'].keys())
        ind_val = list(rekeyed_data_dic[crypto].indicators.values())
        ind_key = list(rekeyed_data_dic[crypto].indicators.keys())  
        
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
        
        if (crypto + interval + 'MA') not in df_dic:
             df_dic[crypto + interval + 'MA'] = \
                 pd.DataFrame([mas_val], index = [scrape_time], columns = [mas_key])
             df_dic[crypto + interval + 'OSC'] = \
                 pd.DataFrame([osc_val], index = [scrape_time], columns = [osc_key])
             df_dic[crypto + interval + 'IND'] = \
                 pd.DataFrame([ind_val], index = [scrape_time], columns = [ind_key])
        else:
            temp_df_dic[crypto + interval + 'MA'] = \
                pd.DataFrame([mas_val], index = [scrape_time], columns = [mas_key])
            temp_df_dic[crypto + interval + 'OSC'] = \
                pd.DataFrame([osc_val], index = [scrape_time], columns = [osc_key])
            temp_df_dic[crypto + interval + 'IND'] = \
                pd.DataFrame([ind_val], index = [scrape_time], columns = [ind_key])
            df_dic[crypto + interval + 'MA'] = \
                df_dic[crypto + interval + 'MA'].append(temp_df_dic[crypto + interval + 'MA'])
            df_dic[crypto + interval + 'OSC'] = \
                df_dic[crypto + interval + 'OSC'].append(temp_df_dic[crypto + interval + 'OSC'])
            df_dic[crypto + interval + 'IND'] = \
                df_dic[crypto + interval + 'IND'].append(temp_df_dic[crypto + interval + 'IND'])
    return df_dic

def format4predicting(alt_df_dic, crypto, trade_interval, pred_intervals, model_features):
    x = []
    for model_num in range(len(pred_intervals)):
        ma_df = alt_df_dic[crypto + pred_intervals[model_num] + 'MA']
        osc_df = alt_df_dic[crypto + pred_intervals[model_num] + 'OSC']
       
        avg_ma  = (ma_df.iloc[-1] + ma_df.iloc[-2]) / 2
        avg_osc = (osc_df.iloc[-1] + osc_df.iloc[-2]) / 2
        inputs = avg_ma.append(avg_osc)
        '''
        if pred_intervals[model_num] == trade_interval:
            ind_df = alt_df_dic[crypto + trade_interval + 'IND'] 
            price = ind_df['close'].iloc[-1, 0]    
        '''
        for header in inputs.index:
            if header[0] not in model_features[model_num]:
                del inputs[header[0]]
        model_inputs = inputs[model_features[model_num]]        
        x.append(np.asarray([model_inputs]))
    # just moved the two below lines to here      
    ind_df = alt_df_dic[crypto + trade_interval + 'IND'] 
    price = ind_df['close'].iloc[-1, 0] 
    
    return x, price

def format_for_training(train_df_dic, time_mode, train_interval, crypto, \
                        file_name, features, first_time, all_x, all_y, all_roi, look_aheads, markov_number, all_numeric_x, window=None, tern=False, price_comp=False):
    ma = train_df_dic[crypto + train_interval + 'MA']
    osc = train_df_dic[crypto + train_interval + 'OSC']
    ind = train_df_dic[crypto + train_interval + 'IND']
    starting_price = train_df_dic[crypto + train_interval + 'IND']['close'].copy()
    
    
    if int(ma.index[-1][-2:]) < 15:
            ma = ma.iloc[:-1].copy()
            osc = osc.iloc[:-1].copy()
            ind = ind.iloc[:-1].copy()
            starting_price = starting_price.iloc[:-1].copy()
            
    if int(ma.index[1][-2:]) < 15:
        ma = ma.iloc[1:].copy()
        osc = osc.iloc[1:].copy()
        ind = ind.iloc[1:].copy()
        starting_price = starting_price.iloc[1:].copy()
    
    assert ma.shape[0] % 2 == 0, 'dataframe incorrect length ' + file_name
    if time_mode == 'Avg':
        ma = ((ma + ma.shift(-1)) / 2)[::2]
        osc = ((osc + osc.shift(-1)) / 2)[::2]
        ind = ((ind + ind.shift(-1)) / 2)[::2]
        s_starting_price = starting_price.shift(periods = -1)
        starting_price = s_starting_price[::2].copy()
        
    elif time_mode == '1st':
        ma = ma[::2]
        osc = osc[::2]
        ind = ind[::2]
        starting_price = starting_price[::2]
        
    else:
        ma  = ma.shift(-1)[::2]
        osc = osc.shift(-1)[::2]
        ind = ind.shift(-1)[::2]
        starting_price = starting_price.shift(-1)[::2]
    ma = ma[::look_aheads]
    osc = osc[::look_aheads]   
    ind = ind[::look_aheads]
    ending_price = starting_price.shift(-1 * look_aheads)
    
    if markov_number == 2:
        ma = ((ma + ma.shift(-1)) / 2)[::2]
        osc = ((osc + osc.shift(-1)) / 2)[::2]
        ind = ((ind + ind.shift(-1)) / 2)[::2]
        starting_price = starting_price.shift(-1)[::2]
        ending_price = ending_price.shift(-1)[::2]
    
    
    ending_price.rename(columns = {'close':'ending_price'}, inplace = True)
    starting_price.rename(columns = {'close':'starting_price'}, inplace = True)
    # puts the x data in a dataframe
    data = pd.concat([ma, osc], axis = 1, join = 'inner')
    # removes unneeded features
    
    for col in data.columns:
        if col[0] not in features:
            del data[col[0]]
    # puts them in correct order
    data = data[features]
    # puts the y data in a dataframe
    ind_price = pd.concat([starting_price, ending_price], axis = 1, join = 'inner')
    ind_price = ind_price[::look_aheads]
    ind_price.dropna(axis = 0, inplace = True)
    
    if not tern:
        ind_price['boo_diff'] = ind_price.apply(lambda l: True if (l['ending_price'] - l['starting_price']) > 0 \
                                else False, axis = 1)
    else:
        ind_price['boo_diff'] = ind_price.apply(lambda l: 2 if ((l['ending_price'] - l['starting_price']) / l['starting_price']) > .001 \
                                else (0 if ((l['ending_price'] - l['starting_price']) / l['starting_price']) < -.001 else 1), axis = 1)
            
    ind_price['diff'] = ind_price.apply(lambda l: (l['ending_price'] - l['starting_price']) / \
                            l['starting_price'] , axis = 1)       
    # np arrays for features and labels (note 2 labels - a boolean up/ down and an actual roi)
   
    
    if price_comp:
       full_features = ind.columns
       pivot_features = full_features[51: 82]
       func = lambda l: l[0]
       pivots = list(map(func, pivot_features))
       
       price_based_features = ['EMA5', 'SMA5', 'EMA10', 'SMA10', 'EMA20', 'SMA20', 'EMA30', 'SMA30', \
                               'EMA50', 'SMA50', 'EMA100', 'SMA100', 'EMA200', 'SMA200', 'Ichimoku.BLine',\
                                   'HullMA9', 'VWMA', 'P.SAR', 'BB.lower', 'BB.upper'] + pivots
       for price_based_feature in price_based_features:
           ind[price_based_feature] = ind.apply(lambda l: l[price_based_feature] - l['close'], axis=1)
    
       
       # additional comparisons
       for name in ['RSI', 'Stoch.K', 'Stoch.D', 'CCI20', 'Mom', 'AO', 'ADX-DI', 'ADX+DI']:
           ind[name] = ind.apply(lambda l: l[name] - l[name + '[1]'], axis=1)
       ind['AO[1]'] = ind.apply(lambda l: l['AO[1]'] - l['AO[2]'], axis=1)
       ind['MACD.macd'] = ind.apply(lambda l: l['MACD.macd'] - l['MACD.signal'], axis=1)
       
    '''
    # TEST STUFFF BELOW
    new = pd.concat(data, ind_price, axis=1, join='inner')
    new['diff'] = new['diff'].replace(0, np.NaN)
    new.dropna(axis = 0, inplace = True)
    
    ################################
    '''
    
    x = np.asarray(data)
    x = x[:-1] 
    
    y = np.asarray(ind_price['boo_diff'])
    roi = np.asarray(ind_price['diff'])
    numeric_x = np.asarray(ind)
    numeric_x = numeric_x[:-1]
    
    
        
    
    if type(window) is int:
       while x.shape[0] % (window) != 0:
           x = x[:-1] 
           numeric_x = numeric_x[:-1]
           roi = roi[:-1]
           y = y[:-1]
    
    cols = ind.columns.values
    # combines all file data together using insert or initializes the arrays if first file
    if first_time:
        all_x = x
        all_y = y
        all_roi = roi
        all_numeric_x = numeric_x
        first_time = False
    else:
        all_x = np.insert(all_x, all_x.shape[0], x, axis = 0)
        all_y = np.insert(all_y, all_y.shape[0], y, axis = 0)
        all_roi = np.insert(all_roi, all_roi.shape[0], roi, axis = 0)
        all_numeric_x = np.insert(all_numeric_x, all_numeric_x.shape[0], numeric_x, axis = 0)
    return all_x, all_y, all_roi, first_time, all_numeric_x, cols




def format_for_multi_training(train_df_dic, time_mode, train_intervals, crypto, \
                        file_name, features, first_time, all_x, all_y, all_roi, \
                            look_aheads, all_numeric_x, window=None, \
                                tern=False, price_comp=False):
    temp_features = []
    the_features = []
    ma = None
    osc = None
    ind = None
    for train_interval in train_intervals:
        
        
        temp_ma = train_df_dic[crypto + train_interval + 'MA']
        temp_osc = train_df_dic[crypto + train_interval + 'OSC']
        temp_ind = train_df_dic[crypto + train_interval + 'IND']
        ma_cols = list(i[0] for i in temp_ma.columns.values)
        osc_cols = list(i[0] for i in temp_osc.columns.values)
        ind_cols = list(i[0] for i in temp_ind.columns.values)
        
        func = lambda l: l + '_' + str(train_interval)
        temp_features = list(map(func, features))
        the_features += temp_features
        new_ma_cols = list(map(func, ma_cols))
        new_osc_cols = list(map(func, osc_cols))
        new_ind_cols = list(map(func, ind_cols))
        
        ma_cols_dic = dict(zip(ma_cols, new_ma_cols))
        osc_cols_dic = dict(zip(osc_cols, new_osc_cols))
        ind_cols_dic = dict(zip(ind_cols, new_ind_cols))
        
        temp_ma.rename(columns=ma_cols_dic, inplace=True)
        temp_osc.rename(columns=osc_cols_dic, inplace=True)
        temp_ind.rename(columns=ind_cols_dic, inplace=True)
        
        '''
        round_idx_func = lambda l: l[:-2] + str(round(int(l[-2:]), -1))
        ma_index = list(temp_ma.index.values)
        osc_index = list(temp_osc.index.values)
        ind_index = list(temp_ind.index.values)
        
        new_ma_idx = list(map(round_idx_func, ma_index))
        new_osc_idx = list(map(round_idx_func, osc_index))
        new_ind_idx = list(map(round_idx_func, ind_index))
        '''
        
        if int(temp_ma.index[-1][-2:]) < 15:
            end_wrong = True
        else:
            end_wrong = False
        if int(temp_ma.index[1][-2:]) < 15:
            start_wrong = True
        else:
            start_wrong = False
            
        
        if train_interval == train_intervals[0]:
            temp_ma.reset_index(inplace=True, drop=False)
        else:
            temp_ma.reset_index(inplace=True, drop=True)
        
        
        temp_osc.reset_index(inplace=True, drop=True)
       
        temp_ind.reset_index(inplace=True)
        
        if ma is not None:
            ma = pd.concat([ma, temp_ma], join='inner', axis=1)
            osc = pd.concat([osc, temp_osc], axis=1)
            ind = pd.concat([ind, temp_ind], axis=1)
        else:
            ma = temp_ma
            osc = temp_osc
            ind = temp_ind
    starting_price = train_df_dic[crypto + train_interval + 'IND']['close_' + train_interval].copy()
    
    
    if end_wrong:
            ma = ma.iloc[:-1].copy()
            osc = osc.iloc[:-1].copy()
            ind = ind.iloc[:-1].copy()
            starting_price = starting_price.iloc[:-1].copy()
            
    if start_wrong:
        ma = ma.iloc[1:].copy()
        osc = osc.iloc[1:].copy()
        ind = ind.iloc[1:].copy()
        starting_price = starting_price.iloc[1:].copy()
    
    assert ma.shape[0] % 2 == 0, 'dataframe incorrect length ' + file_name
    if time_mode == 'Avg':
        ma = ((ma + ma.shift(-1)) / 2)[::2]
        osc = ((osc + osc.shift(-1)) / 2)[::2]
        ind = ((ind + ind.shift(-1)) / 2)[::2]
        s_starting_price = starting_price.shift(periods = -1)
        starting_price = s_starting_price[::2].copy()
        
    elif time_mode == '1st':
        ma = ma[::2]
        osc = osc[::2]
        ind = ind[::2]
        starting_price = starting_price[::2]
        
    else:
        ma  = ma.shift(-1)[::2]
        osc = osc.shift(-1)[::2]
        ind = ind.shift(-1)[::2]
        starting_price = starting_price.shift(-1)[::2]
   
    ending_price = starting_price.shift(-1 * look_aheads)
    

    
    ending_price.rename(columns = {'close_' + train_interval:'ending_price'}, inplace = True)
    starting_price.rename(columns = {'close_' + train_interval:'starting_price'}, inplace = True)
    # puts the x data in a dataframe
    data = pd.concat([ma, osc], axis = 1, join = 'inner')
    # removes unneeded features
    
    for col in data.columns:
        if col[0] not in the_features:
            del data[col[0]]
    # puts them in correct order
    data = data[the_features]
    # puts the y data in a dataframe
    ind_price = pd.concat([starting_price, ending_price], axis = 1, join = 'inner')
    ind_price.dropna(axis = 0, inplace = True)
    
    if not tern:
        ind_price['boo_diff'] = ind_price.apply(lambda l: True if (l['ending_price'] - l['starting_price']) > 0 \
                                else False, axis = 1)
    else:
        ind_price['boo_diff'] = ind_price.apply(lambda l: 2 if ((l['ending_price'] - l['starting_price']) / l['starting_price']) > .001 \
                                else (0 if ((l['ending_price'] - l['starting_price']) / l['starting_price']) < -.001 else 1), axis = 1)
            
    ind_price['diff'] = ind_price.apply(lambda l: (l['ending_price'] - l['starting_price']) / \
                            l['starting_price'] , axis = 1)       
   
   
    # when x is using indicators as continunous values - scaling them relative to current price
    if price_comp:
       full_features = ind.columns
       pivot_features = full_features[51: 82]
       func = lambda l: l[0]
       pivots = list(map(func, pivot_features))
       
       price_based_features = ['EMA5', 'SMA5', 'EMA10', 'SMA10', 'EMA20', 'SMA20', 'EMA30', 'SMA30', \
                               'EMA50', 'SMA50', 'EMA100', 'SMA100', 'EMA200', 'SMA200', 'Ichimoku.BLine',\
                                   'HullMA9', 'VWMA', 'P.SAR', 'BB.lower', 'BB.upper'] + pivots
       for price_based_feature in price_based_features:
           ind[price_based_feature] = ind.apply(lambda l: l[price_based_feature] - l['close'], axis=1)
       # additional comparisons
       for name in ['RSI', 'Stoch.K', 'Stoch.D', 'CCI20', 'Mom', 'AO', 'ADX-DI', 'ADX+DI']:
           ind[name] = ind.apply(lambda l: l[name] - l[name + '[1]'], axis=1)
       ind['AO[1]'] = ind.apply(lambda l: l['AO[1]'] - l['AO[2]'], axis=1)
       ind['MACD.macd'] = ind.apply(lambda l: l['MACD.macd'] - l['MACD.signal'], axis=1)
 
    
 
    x = np.asarray(data)
    x = x[:-1 * look_aheads] 
    y = np.asarray(ind_price['boo_diff'])
    roi = np.asarray(ind_price['diff'])
    numeric_x = np.asarray(ind)
    numeric_x = numeric_x[:-1 * look_aheads]
   
    x = x[::look_aheads]
    y = y[::look_aheads]
    roi = roi[::look_aheads]
    numeric_x = numeric_x[::look_aheads]
    
    
    # trims off ending of x and y so it can reshaped to 3d array for RNN
    if type(window) is int:
       while x.shape[0] % (window) != 0:
           x = x[:-1] 
           numeric_x = numeric_x[:-1]
           roi = roi[:-1]
           y = y[:-1]
    
    cols = ind.columns.values
    # combines all file data together using insert or initializes the arrays if first file
    

    
    if first_time:
        all_x = x
        all_y = y
        all_roi = roi
        all_numeric_x = numeric_x
        first_time = False
    else:
        all_x = np.insert(all_x, all_x.shape[0], x, axis = 0)
        all_y = np.insert(all_y, all_y.shape[0], y, axis = 0)
        all_roi = np.insert(all_roi, all_roi.shape[0], roi, axis = 0)
        all_numeric_x = np.insert(all_numeric_x, all_numeric_x.shape[0], numeric_x, axis = 0)
    return all_x, all_y, all_roi, first_time, all_numeric_x, cols




 
def format_for_csv(all_roi, all_x, features, bool_roi):
    if not bool_roi:
        all_roi *= 100
    new_roi = all_roi.reshape((-1, 1))
    all_data = np.concatenate((new_roi, all_x), axis=1)
    if not bool_roi:
        df = pd.DataFrame(all_data, columns=['% ROI'] + features)
    else:
        df = pd.DataFrame(all_data, columns=['Price Up?'] + features)
    return df
    
        
def format_for_rl_pred(x):
    
    for idx in range(len(x)):
        
        if x[idx] == .5:
           x[idx] = np.random.choice([0, 1])
        elif x[idx] == -.5:
            x[idx] == np.random.choice([0,-1])
        x[idx] = int(x[idx])
    
    state_str = ''.join(str(item) for item in x)
    state_str = state_str.replace('.0', '')
    
    return state_str           
        
            
        
        
        