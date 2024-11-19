
import ast
import os
import re
from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils import compute_class_weight
from tqdm.auto import tqdm
from utils import *

class DataGenerator:
    def __init__(self, data_path='./stock_history'):
        self.strategy_type = 'original'
        self.data_path = data_path
        self.output_path = os.path.join(data_path, f"features_{data_path}.csv")
        self.BASE_URL = ""  # api key from alpha vantage service
        self.start_col = 'open'
        self.end_col = 'eom_200'
        self.df = self.create_features()
        self.one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
        self.one_hot_enc.fit(self.df['labels'].values.reshape(-1, 1))
        self.batch_start_date = self.df.head(1).iloc[0]["timestamp"]
        self.test_duration_years = 1

    def calculate_technical_indicators(self, df, col_name, intervals):
        get_RSI_smooth(df, col_name, intervals)  # momentum
        get_williamR(df, col_name, intervals)  # momentum
        # get_mfi(df, intervals)  # momentum
        get_ROC(df, col_name, intervals)  # momentum
        # get_CMF(df, col_name, intervals)  # momentum
        get_CMO(df, col_name, intervals)  # momentum
        get_SMA(df, col_name, intervals)
        get_SMA(df, 'open', intervals)
        get_EMA(df, col_name, intervals)
        get_WMA(df, col_name, intervals)
        get_HMA(df, col_name, intervals)
        get_macd(df)
        get_adx(df, col_name, intervals)  # trend
        get_TRIX(df, col_name, intervals)  # trend
        get_CCI(df, col_name, intervals)  # trend
        get_DPO(df, col_name, intervals)  # Trend oscillator
        get_kst(df, col_name, intervals)  # Trend
        get_DMI(df, col_name, intervals)  # trend
        get_BB_MAV(df, col_name, intervals)  # volatility
        # get_force_index(df, intervals)  # volume
        # get_kdjk_rsv(df, intervals)
        # get_OBV(df)  # volume
        # get_EOM(df, col_name, intervals)  # volume momentum
        # get_volume_delta(df)  # volume
        # get_IBR(df)

    def create_labels_price(self, df, col_name, window_size=12):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels with
                  size = total-(window_size)+1
        """

        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)

        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter

                high_window = df.loc[window_begin : window_end]['high']
                low_window = df.loc[window_begin : window_end]['low']

                max_index = high_window.idxmax()
                min_index = low_window.idxmin()
                max_ = high_window.max()
                min_ = low_window.min()

                current_price = df.loc[window_begin, col_name]

                if (max_ - current_price > 7) and (current_price - min_ > 7):
                    if max_index < min_index:
                        min_low_small = df.loc[window_begin : max_index]['low'].min()
                        if current_price - min_low_small < 3:
                            labels[window_begin] = 1
                        else:
                            labels[window_begin] = 2
                    else:
                        max_high_small = df.loc[window_begin : min_index]['high'].max()
                        if max_high_small - current_price < 3:
                            labels[window_begin] = 0
                        else:
                            labels[window_begin] = 2
                elif (max_ - current_price > 7) and (current_price - min_ <= 7):
                    min_low_small = df.loc[window_begin : max_index]['low'].min()
                    if current_price - min_low_small < 3:
                        labels[window_begin] = 1
                    else:
                        labels[window_begin] = 2
                elif (max_ - current_price <= 7) and (current_price - min_ > 7):
                    max_high_small = df.loc[window_begin : min_index]['high'].max()
                    if max_high_small - current_price < 3:
                        labels[window_begin] = 0
                    else:
                        labels[window_begin] = 2
                else:
                    labels[window_begin] = 2
                  
            row_counter = row_counter + 1
            pbar.update(1)

        pbar.close()
        return labels

    def create_labels(self, df, col_name, window_size=11):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels with
                  size = total-(window_size)+1
        """

        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)

        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = int((window_begin + window_end) / 2)

                high_window = df.loc[window_begin : window_end]['high']
                low_window = df.loc[window_begin : window_end]['low']

                max_index = high_window.idxmax()
                min_index = low_window.idxmin()
                max_ = high_window.max()
                min_ = low_window.min()

                min_after = df.loc[max_index : window_end]['low'].min()
                max_after = df.loc[min_index : window_end]['high'].max()

                current_price = df.iloc[window_middle]['close']

                if max_index == window_middle and current_price - min_after > 5:
                    labels[window_middle] = 0
                elif min_index == window_middle and max_after - current_price > 5:
                    labels[window_middle] = 1
                else:
                    labels[window_middle] = 2

            row_counter = row_counter + 1
            pbar.update(1)

        pbar.close()
        return labels

    def create_labels_price_rise(self, df, col_name):
        """
        labels data based on price rise on next day
          next_day - prev_day
        ((s - s.shift()) > 0).astype(np.int)
        """

        df["labels"] = ((df[col_name] - df[col_name].shift()) > 0).astype(np.int)
        df = df[1:]
        df.reset_index(drop=True, inplace=True)

    def create_label_mean_reversion(self, df, col_name):
        """
        strategy as described at "https://decodingmarkets.com/mean-reversion-trading-strategy"

        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels
        """

        get_RSI_smooth(df, col_name, [3])  # new column 'rsi_3' added to df
        rsi_3_series = df['rsi_3']
        ibr = get_IBR(df)
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        count = 0
        for i, rsi_3 in enumerate(rsi_3_series):
            if rsi_3 < 15:  # buy
                count = count + 1

                if 3 <= count < 8 and ibr.iloc[i] < 0.2:  # TODO implement upto 5 BUYS
                    labels[i] = 1

                if count >= 8:
                    count == 0
            elif ibr.iloc[i] > 0.7:  # sell
                labels[i] = 0
            else:
                labels[i] = 2

        return labels

    def create_label_short_long_ma_crossover(self, df, col_name, short, long):
        """
        if short = 30 and long = 90,
        Buy when 30 day MA < 90 day MA
        Sell when 30 day MA > 90 day MA

        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels
        """

        def detect_crossover(diff_prev, diff):
            if diff_prev >= 0 > diff:
                # buy
                return 1
            elif diff_prev <= 0 < diff:
                return 0
            else:
                return 2

        get_SMA(df, 'close', [short, long])
        labels = np.zeros((len(df)))
        labels[:] = np.nan
        diff = df['close_sma_' + str(short)] - df['close_sma_' + str(long)]
        diff_prev = diff.shift()
        df['diff_prev'] = diff_prev
        df['diff'] = diff

        res = df.apply(lambda row: detect_crossover(row['diff_prev'], row['diff']), axis=1)
        print("labels count", np.unique(res, return_counts=True))
        df.drop(columns=['diff_prev', 'diff'], inplace=True)
        return res
    
    def get_trend(self, df, technical_info):
        technical_data = technical_info.values
        for i in range(len(technical_data)):
            trend_data = ast.literal_eval(technical_data[i][0]).get('current', [])
            if trend_data:
                if trend_data[0]['trend_name'] is not None:
                    trend_name = 0 if 'down' in trend_data[0]['trend_name'] else 1
                    durration_trend = trend_data[0].get('duration_trend', 0)
                    number_pullback = trend_data[0].get('number_pullback', 0)
                else:
                    trend_name = 2
                    durration_trend = 0
                    number_pullback = 0
            else:
                trend_name = 2
                durration_trend = 0
                number_pullback = 0

            df.at[i, 'trend_name'] = trend_name
            df.at[i, 'durration_trend'] = durration_trend
            df.at[i, 'number_pullback'] = number_pullback

            # td, ha
            td_seq_ha = ast.literal_eval(technical_data[i][0]).get('td_sequential_ha', None)
            td_seq = ast.literal_eval(technical_data[i][0]).get('td_sequential', None)

            if td_seq_ha == 0:
                df.at[i, 'td_seq_ha_trend'] = 0
                df.at[i, 'td_seq_ha_number'] = 0
            else:
                df.at[i, 'td_seq_ha_number'] = int(td_seq_ha[0])
                df.at[i, 'td_seq_ha_trend'] = 0 if 'up' in td_seq_ha else 1

            # Resistance
            list_resistances = ast.literal_eval(technical_data[i][0]).get('resistances_list', None)
            if len(list_resistances) > 0:
                resistance = list_resistances[0]
                y_resistance_max = max(resistance[1], resistance[2])
                y_resistance_min = min(resistance[1], resistance[2])
                number_touch_resistance = resistance[3]
                count_candle_touch_resistance = sum(resistance[4])
            else:
                y_resistance_max = df.loc[i]["high"]
                y_resistance_min = df.loc[i]["high"]
                number_touch_resistance = 1
                count_candle_touch_resistance = 1

            # Support
            list_supports = ast.literal_eval(technical_data[i][0]).get('supports_list', None)
            if len(list_supports) > 0:
                support = list_supports[0]
                y_support_max = min(support[1], support[2])
                y_support_min = max(support[1], support[2])
                number_touch_support = support[3]
                count_candle_touch_support = sum(support[4])
            else:
                y_support_min = df.loc[i]["low"]
                y_support_max = df.loc[i]["low"]
                number_touch_support = 1
                count_candle_touch_support = 1

            df.at[i, 'y_resistance_max'] = y_resistance_max
            df.at[i, 'y_support_max'] = y_support_max
            df.at[i, 'y_resistance_min'] = y_resistance_min
            df.at[i, 'y_support_min'] = y_support_min
        return df
                
    def create_features(self):
        if not os.path.exists(self.output_path):
            csv_files_train = [f for f in os.listdir(self.data_path) if f.endswith('.csv') and 'features' not in f]
            df_raw = pd.concat([pd.read_csv(os.path.join(self.data_path, f)) for f in csv_files_train])
            df = df_raw[['Date', 'Open', 'High', 'Low', 'Close', 'volume']]
            df.rename(columns={'Close': 'close', 'Open': 'open', 'Date': 'timestamp', 'High': 'high', 'Low': 'low'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # trend
            technical_info = df_raw[['technical_info']]
            # df = self.get_trend(df, technical_info)

            intervals = [3, 6, 7, 9, 10, 13, 14, 17, 21, 25, 34, 50, 89, 100, 200]
            self.calculate_technical_indicators(df, 'close', intervals)
            
            df.to_csv(self.output_path, index=False)
        else:
            df = pd.read_csv(self.output_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

        prev_len = len(df)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        if 'labels' not in df.columns:
            if re.match(r"\d+_\d+_ma", self.strategy_type):
                short = self.strategy_type.split('_')[0]
                long = self.strategy_type.split('_')[1]
                df['labels'] = self.create_label_short_long_ma_crossover(df, 'close', short, long)
            else:
                df['labels'] = self.create_labels(df, 'close')
                # df['labels'] = self.create_labels_price(df, 'close')

            prev_len = len(df)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(self.output_path, index=False)
        else:
            print("labels already calculated")
        return df
