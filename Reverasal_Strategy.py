import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as dr
from collections import Counter, OrderedDict



def get_data(spsymbols, start, end):
    dict_of_dfs = {}
    for symbol in spsymbols[48:53]:
        print(symbol)
        data = dr.data.get_data_yahoo(symbol, start, end)
        dict_of_dfs[symbol] = data
    return dict_of_dfs






def add_candle_sign(df):
    ''' label each candle as up or down depending on the difference between open and close'''
    candle_sign = []
    for i in range(df.shape[0]):
        Open = df.iloc[i:i+1, 3].values.tolist()[0]
        close = df.iloc[i:i+1, 4].values.tolist()[0]
        if Open > close:
            candle_sign.append('Down')
        elif close > Open:
            candle_sign.append('Up')
        else:
            candle_sign.append('Neutral')
    df['candle_sign'] = candle_sign
    return df





# trade_type means short or long
def is_trade_successful(df, start_ind, trade_type, risk_reward_ratio=3):
    result = 0
    days_in_trade = 0
    i = start_ind
    if trade_type == 'Long':
        entry_price = df.loc[start_ind][1]
        stop_price = df.loc[start_ind][2]
        stop_size = abs(entry_price - stop_price)

    if trade_type == 'Short':
        entry_price = df.loc[start_ind][2]
        stop_price = df.loc[start_ind][1]
        stop_size = abs(entry_price - stop_price)

    if trade_type == 'Long':
        while i < df.shape[0]:
            if df.loc[i][2] < stop_price:
                days_in_trade = i - start_ind
                break
            elif (df.loc[i][1] - entry_price) >= (stop_size * risk_reward_ratio):
                result = 1
                days_in_trade = i - start_ind
                break
            else:
                i += 1

    else:
        while i < df.shape[0]:
            if df.loc[i][1] > stop_price:
                days_in_trade = i - start_ind
                break
            elif abs(df.loc[i][2] - entry_price) >= (
                    stop_size * risk_reward_ratio):  # this is a short trade hence I take the absolute value
                result = 1
                days_in_trade = i - start_ind
                break
            else:
                i += 1

    if days_in_trade == 0:  # if we still in the trade
        days_in_trade = -1

    return result, days_in_trade



def add_trade(trade_dir, curr_ind, df, start_ind, risk_reward_ratio):
    global date
    global entry_price
    global stop_price
    global t_High
    global t_Low
    global t_Open
    global t_Close
    global t_Volume
    global t_Adj_Close
    global t_candle_sign
    global t_minus_1_High
    global t_minus_1_Low
    global t_minus_1_Open
    global t_minus_1_Close
    global t_minus_1_Volume
    global t_minus_1_Adj_Close
    global t_minus_1_candle_sign
    global t_minus_2_High
    global t_minus_2_Low
    global t_minus_2_Open
    global t_minus_2_Close
    global t_minus_2_Volume
    global t_minus_2_Adj_Close
    global t_minus_2_candle_sign
    global t_minus_3_High
    global t_minus_3_Low
    global t_minus_3_Open
    global t_minus_3_Close
    global t_minus_3_Volume
    global t_minus_3_Adj_Close
    global t_minus_3_candle_sign
    global days_in_trade
    global is_successful_trade

    if len(date) == 0:
        if trade_dir == 'Up':
            date.append(df.loc[curr_ind][0])
            entry_price.append(df.loc[curr_ind][1])
            stop_price.append(df.loc[curr_ind][2])
            t_High.append(df.loc[curr_ind][1])
            t_Low.append(df.loc[curr_ind][2])
            t_Open.append(df.loc[curr_ind][3])
            t_Close.append(df.loc[curr_ind][4])
            t_Volume.append(df.loc[curr_ind][5])
            t_Adj_Close.append(df.loc[curr_ind][6])
            t_candle_sign.append(df.loc[curr_ind][7])
            t_minus_1_High.append(df.loc[curr_ind - 1][1])
            t_minus_1_Low.append(df.loc[curr_ind - 1][2])
            t_minus_1_Open.append(df.loc[curr_ind - 1][3])
            t_minus_1_Close.append(df.loc[curr_ind - 1][4])
            t_minus_1_Volume.append(df.loc[curr_ind - 1][5])
            t_minus_1_Adj_Close.append(df.loc[curr_ind - 1][6])
            t_minus_1_candle_sign.append(df.loc[curr_ind - 1][7])
            t_minus_2_High.append(df.loc[curr_ind - 2][1])
            t_minus_2_Low.append(df.loc[curr_ind - 2][2])
            t_minus_2_Open.append(df.loc[curr_ind - 2][3])
            t_minus_2_Close.append(df.loc[curr_ind - 2][4])
            t_minus_2_Volume.append(df.loc[curr_ind - 2][5])
            t_minus_2_Adj_Close.append(df.loc[curr_ind - 2][6])
            t_minus_2_candle_sign.append(df.loc[curr_ind - 2][7])
            t_minus_3_High.append(df.loc[curr_ind - 3][1])
            t_minus_3_Low.append(df.loc[curr_ind - 3][2])
            t_minus_3_Open.append(df.loc[curr_ind - 3][3])
            t_minus_3_Close.append(df.loc[curr_ind - 3][4])
            t_minus_3_Volume.append(df.loc[curr_ind - 3][5])
            t_minus_3_Adj_Close.append(df.loc[curr_ind - 3][6])
            t_minus_3_candle_sign.append(df.loc[curr_ind - 3][7])
            result, trade_duration = is_trade_successful(df, start_ind, "Long", risk_reward_ratio)
            days_in_trade.append(trade_duration)
            is_successful_trade.append(result)
            curr_ind = start_ind + trade_duration

        if trade_dir == 'Down':
            date.append(df.loc[curr_ind][0])
            entry_price.append(df.loc[curr_ind][2])
            stop_price.append(df.loc[curr_ind][1])
            t_High.append(df.loc[curr_ind][1])
            t_Low.append(df.loc[curr_ind][2])
            t_Open.append(df.loc[curr_ind][3])
            t_Close.append(df.loc[curr_ind][4])
            t_Volume.append(df.loc[curr_ind][5])
            t_Adj_Close.append(df.loc[curr_ind][6])
            t_candle_sign.append(df.loc[curr_ind][7])
            t_minus_1_High.append(df.loc[curr_ind - 1][1])
            t_minus_1_Low.append(df.loc[curr_ind - 1][2])
            t_minus_1_Open.append(df.loc[curr_ind - 1][3])
            t_minus_1_Close.append(df.loc[curr_ind - 1][4])
            t_minus_1_Volume.append(df.loc[curr_ind - 1][5])
            t_minus_1_Adj_Close.append(df.loc[curr_ind - 1][6])
            t_minus_1_candle_sign.append(df.loc[curr_ind - 1][7])
            t_minus_2_High.append(df.loc[curr_ind - 2][1])
            t_minus_2_Low.append(df.loc[curr_ind - 2][2])
            t_minus_2_Open.append(df.loc[curr_ind - 2][3])
            t_minus_2_Close.append(df.loc[curr_ind - 2][4])
            t_minus_2_Volume.append(df.loc[curr_ind - 2][5])
            t_minus_2_Adj_Close.append(df.loc[curr_ind - 2][6])
            t_minus_2_candle_sign.append(df.loc[curr_ind - 2][7])
            t_minus_3_High.append(df.loc[curr_ind - 3][1])
            t_minus_3_Low.append(df.loc[curr_ind - 3][2])
            t_minus_3_Open.append(df.loc[curr_ind - 3][3])
            t_minus_3_Close.append(df.loc[curr_ind - 3][4])
            t_minus_3_Volume.append(df.loc[curr_ind - 3][5])
            t_minus_3_Adj_Close.append(df.loc[curr_ind - 3][6])
            t_minus_3_candle_sign.append(df.loc[curr_ind - 3][7])
            result, trade_duration = is_trade_successful(df, start_ind, "Short", risk_reward_ratio)
            days_in_trade.append(trade_duration)
            is_successful_trade.append(result)
            curr_ind = start_ind + trade_duration

        if curr_ind == start_ind:
            curr_ind += 1

    if df.loc[start_ind][0] > date[-1] and len(date) > 0:
        if trade_dir == 'Up':
            date.append(df.loc[curr_ind][0])
            entry_price.append(df.loc[curr_ind][1])
            stop_price.append(df.loc[curr_ind][2])
            t_High.append(df.loc[curr_ind][1])
            t_Low.append(df.loc[curr_ind][2])
            t_Open.append(df.loc[curr_ind][3])
            t_Close.append(df.loc[curr_ind][4])
            t_Volume.append(df.loc[curr_ind][5])
            t_Adj_Close.append(df.loc[curr_ind][6])
            t_candle_sign.append(df.loc[curr_ind][7])
            t_minus_1_High.append(df.loc[curr_ind - 1][1])
            t_minus_1_Low.append(df.loc[curr_ind - 1][2])
            t_minus_1_Open.append(df.loc[curr_ind - 1][3])
            t_minus_1_Close.append(df.loc[curr_ind - 1][4])
            t_minus_1_Volume.append(df.loc[curr_ind - 1][5])
            t_minus_1_Adj_Close.append(df.loc[curr_ind - 1][6])
            t_minus_1_candle_sign.append(df.loc[curr_ind - 1][7])
            t_minus_2_High.append(df.loc[curr_ind - 2][1])
            t_minus_2_Low.append(df.loc[curr_ind - 2][2])
            t_minus_2_Open.append(df.loc[curr_ind - 2][3])
            t_minus_2_Close.append(df.loc[curr_ind - 2][4])
            t_minus_2_Volume.append(df.loc[curr_ind - 2][5])
            t_minus_2_Adj_Close.append(df.loc[curr_ind - 2][6])
            t_minus_2_candle_sign.append(df.loc[curr_ind - 2][7])
            t_minus_3_High.append(df.loc[curr_ind - 3][1])
            t_minus_3_Low.append(df.loc[curr_ind - 3][2])
            t_minus_3_Open.append(df.loc[curr_ind - 3][3])
            t_minus_3_Close.append(df.loc[curr_ind - 3][4])
            t_minus_3_Volume.append(df.loc[curr_ind - 3][5])
            t_minus_3_Adj_Close.append(df.loc[curr_ind - 3][6])
            t_minus_3_candle_sign.append(df.loc[curr_ind - 3][7])
            result, trade_duration = is_trade_successful(df, start_ind, "Long", risk_reward_ratio)
            days_in_trade.append(trade_duration)
            is_successful_trade.append(result)
            curr_ind = start_ind + trade_duration

        if trade_dir == 'Down':
            date.append(df.loc[curr_ind][0])
            entry_price.append(df.loc[curr_ind][2])
            stop_price.append(df.loc[curr_ind][1])
            t_High.append(df.loc[curr_ind][1])
            t_Low.append(df.loc[curr_ind][2])
            t_Open.append(df.loc[curr_ind][3])
            t_Close.append(df.loc[curr_ind][4])
            t_Volume.append(df.loc[curr_ind][5])
            t_Adj_Close.append(df.loc[curr_ind][6])
            t_candle_sign.append(df.loc[curr_ind][7])
            t_minus_1_High.append(df.loc[curr_ind - 1][1])
            t_minus_1_Low.append(df.loc[curr_ind - 1][2])
            t_minus_1_Open.append(df.loc[curr_ind - 1][3])
            t_minus_1_Close.append(df.loc[curr_ind - 1][4])
            t_minus_1_Volume.append(df.loc[curr_ind - 1][5])
            t_minus_1_Adj_Close.append(df.loc[curr_ind - 1][6])
            t_minus_1_candle_sign.append(df.loc[curr_ind - 1][7])
            t_minus_2_High.append(df.loc[curr_ind - 2][1])
            t_minus_2_Low.append(df.loc[curr_ind - 2][2])
            t_minus_2_Open.append(df.loc[curr_ind - 2][3])
            t_minus_2_Close.append(df.loc[curr_ind - 2][4])
            t_minus_2_Volume.append(df.loc[curr_ind - 2][5])
            t_minus_2_Adj_Close.append(df.loc[curr_ind - 2][6])
            t_minus_2_candle_sign.append(df.loc[curr_ind - 2][7])
            t_minus_3_High.append(df.loc[curr_ind - 3][1])
            t_minus_3_Low.append(df.loc[curr_ind - 3][2])
            t_minus_3_Open.append(df.loc[curr_ind - 3][3])
            t_minus_3_Close.append(df.loc[curr_ind - 3][4])
            t_minus_3_Volume.append(df.loc[curr_ind - 3][5])
            t_minus_3_Adj_Close.append(df.loc[curr_ind - 3][6])
            t_minus_3_candle_sign.append(df.loc[curr_ind - 3][7])
            result, trade_duration = is_trade_successful(df, start_ind, "Short", risk_reward_ratio)
            days_in_trade.append(trade_duration)
            is_successful_trade.append(result)
            curr_ind = start_ind + trade_duration

        if curr_ind == start_ind:
            curr_ind += 1
    else:
        curr_ind = start_ind + 1

    return curr_ind


# since the basic rules define at least 4 candles we will store the data from the 4 last days as the basic features for the trade
def generate_trades(df, risk_reward_ratio=3):
    optional_trades = pd.DataFrame()

    for i in range(5, df.shape[0] - 10):
        current_sign = df.loc[i][-1]
        t_minus_1_sign = df.loc[i - 1][-1]
        t_minus_2_sign = df.loc[i - 2][-1]
        t_minus_3_sign = df.loc[i - 3][-1]
        t_minus_4_sign = df.loc[i - 4][-1]
        t_minus_5_sign = df.loc[i - 5][-1]

        trade_direction = ''
        if current_sign == 'Up':
            trade_direction = 'Down'
        else:
            trade_direction = 'Up'
        # check first rule - 4 candles in the same direction
        if (current_sign == t_minus_1_sign == t_minus_2_sign == t_minus_3_sign) and (current_sign != 'Neutral'):
            for j in range(1, 6):  # a trade can enter only in range of 5 days
                ind = i + j
                if trade_direction == 'Up':
                    if (df.loc[ind][3] <= df.loc[i][1] and df.loc[ind][3] >= df.loc[i][2]) and (
                            df.loc[ind][1] > df.loc[i][1]):

                        i = add_trade(trade_direction, i, df, ind, risk_reward_ratio)
                        break

                if trade_direction == 'Down':
                    if (df.loc[ind][3] <= df.loc[i][1] and df.loc[ind][3] >= df.loc[i][2]) and (
                            df.loc[ind][2] < df.loc[i][2]):
                        i = add_trade(trade_direction, i, df, ind, risk_reward_ratio)
                        break

                        # check second rule - 4 out of 5 candles are in the same direction
        last_5_candle_signs = [current_sign, t_minus_1_sign, t_minus_2_sign, t_minus_3_sign, t_minus_4_sign]
        c = dict(Counter(last_5_candle_signs))
        is_4_out_of_5 = 1 if 4 in list(c.values()) else 0
        max_value = max(list(c.values()))
        max_index = list(c.values()).index(max_value)
        majority_sign = list(c.keys())[max_index]

        if (is_4_out_of_5 == 1) and (current_sign == majority_sign):  # stopped here
            for j in range(1, 6):  # a trade can enter only in range of 5 days
                ind = i + j
                if (trade_direction == 'Up') and (df.loc[ind][3] <= df.loc[i][1]) and (
                        df.loc[ind][3] >= df.loc[i][2]) and (df.loc[ind][1] > df.loc[i][1]):
                    i = add_trade(trade_direction, i, df, ind, risk_reward_ratio)
                    break

                if (trade_direction == 'Down') and (df.loc[ind][3] <= df.loc[i][1]) and (
                        df.loc[ind][3] >= df.loc[i][2]) and (df.loc[ind][2] < df.loc[i][2]):
                    i = add_trade(trade_direction, i, df, ind, risk_reward_ratio)
                    break

        # check third rule - 5 out of 6 candles are in the same direction
        last_6_candle_signs = [current_sign, t_minus_1_sign, t_minus_2_sign, t_minus_3_sign, t_minus_4_sign,
                               t_minus_5_sign]
        c = dict(Counter(last_5_candle_signs))
        is_5_out_of_6 = 1 if 5 in list(c.values()) else 0
        max_value = max(list(c.values()))
        max_index = list(c.values()).index(max_value)
        majority_sign = list(c.keys())[max_index]

        if (is_5_out_of_6 == 1) and (current_sign == majority_sign):  # stopped here
            for j in range(1, 6):  # a trade can enter only in range of 5 days
                ind = i + j
                if (trade_direction == 'Up') and (df.loc[ind][3] <= df.loc[i][1]) and (
                        df.loc[ind][3] >= df.loc[i][2]) and (df.loc[ind][1] > df.loc[i][1]):
                    i = add_trade(trade_direction, i, df, ind, risk_reward_ratio)
                    break

                if (trade_direction == 'Down') and (df.loc[ind][3] <= df.loc[i][1]) and (
                        df.loc[ind][3] >= df.loc[i][2]) and (df.loc[ind][2] < df.loc[i][2]):
                    i = add_trade(trade_direction, i, df, ind, risk_reward_ratio)
                    break

    optional_trades['date'] = date
    optional_trades['entry_price'] = entry_price
    optional_trades['stop_price'] = stop_price
    optional_trades['t_High'] = t_High
    optional_trades['t_Low'] = t_Low
    optional_trades['t_Open'] = t_Open
    optional_trades['t_Close'] = t_Close
    optional_trades['t_Volume'] = t_Volume
    optional_trades['t_Adj_Close'] = t_Adj_Close
    optional_trades['t_candle_sign'] = t_candle_sign
    optional_trades['t_minus_1_High'] = t_minus_1_High
    optional_trades['t_minus_1_Low'] = t_minus_1_Low
    optional_trades['t_minus_1_Open'] = t_minus_1_Open
    optional_trades['t_minus_1_Close'] = t_minus_1_Close
    optional_trades['t_minus_1_Volume'] = t_minus_1_Volume
    optional_trades['t_minus_1_Adj_Close'] = t_minus_1_Adj_Close
    optional_trades['t_minus_1_candle_sign'] = t_minus_1_candle_sign
    optional_trades['t_minus_2_High'] = t_minus_2_High
    optional_trades['t_minus_2_Low'] = t_minus_2_Low
    optional_trades['t_minus_2_Open'] = t_minus_2_Open
    optional_trades['t_minus_2_Close'] = t_minus_2_Close
    optional_trades['t_minus_2_Volume'] = t_minus_2_Volume
    optional_trades['t_minus_2_Adj_Close'] = t_minus_2_Adj_Close
    optional_trades['t_minus_2_candle_sign'] = t_minus_2_candle_sign
    optional_trades['t_minus_3_High'] = t_minus_3_High
    optional_trades['t_minus_3_Low'] = t_minus_3_Low
    optional_trades['t_minus_3_Open'] = t_minus_3_Open
    optional_trades['t_minus_3_Close'] = t_minus_3_Close
    optional_trades['t_minus_3_Volume'] = t_minus_3_Volume
    optional_trades['t_minus_3_Adj_Close'] = t_minus_3_Adj_Close
    optional_trades['t_minus_3_candle_sign'] = t_minus_3_candle_sign
    optional_trades['days_in_trade'] = days_in_trade
    optional_trades['is_successful_trade'] = is_successful_trade

    return optional_trades


if __name__ == "__main__":
    data = dr.data.get_data_yahoo('AAPL', '2005-09-27', '2019-12-25')
    data = data.reset_index()
    data = add_candle_sign(data)
    date = []
    entry_price = []
    stop_price = []
    t_High = []
    t_Low = []
    t_Open = []
    t_Close = []
    t_Volume = []
    t_Adj_Close = []
    t_candle_sign = []
    t_minus_1_High = []
    t_minus_1_Low = []
    t_minus_1_Open = []
    t_minus_1_Close = []
    t_minus_1_Volume = []
    t_minus_1_Adj_Close = []
    t_minus_1_candle_sign = []
    t_minus_2_High = []
    t_minus_2_Low = []
    t_minus_2_Open = []
    t_minus_2_Close = []
    t_minus_2_Volume = []
    t_minus_2_Adj_Close = []
    t_minus_2_candle_sign = []
    t_minus_3_High = []
    t_minus_3_Low = []
    t_minus_3_Open = []
    t_minus_3_Close = []
    t_minus_3_Volume = []
    t_minus_3_Adj_Close = []
    t_minus_3_candle_sign = []
    days_in_trade = []
    is_successful_trade = []
    print(data.head())
    trades = generate_trades(data, risk_reward_ratio=3)
    trades.to_csv('reversal_trades_AAPL3.csv')