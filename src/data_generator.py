
import ast
import os
import re
from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
        self.end_col = 'eom_26'
        self.df = self.create_features()
        self.feat_idx = self.feature_selection()
        self.one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
        self.one_hot_enc.fit(self.df['labels'].values.reshape(-1, 1))
        self.batch_start_date = self.df.head(1).iloc[0]["timestamp"]
        self.test_duration_years = 1

    def calculate_technical_indicators(self, df, col_name, intervals):
        # get_RSI(df, col_name, intervals)  # faster but non-smoothed RSI
        get_RSI_smooth(df, col_name, intervals)  # momentum
        get_williamR(df, col_name, intervals)  # momentum
        get_mfi(df, intervals)  # momentum
        # get_MACD(df, col_name, intervals)  # momentum, ready to use +3
        # get_PPO(df, col_name, intervals)  # momentum, ready to use +1
        get_ROC(df, col_name, intervals)  # momentum
        get_CMF(df, col_name, intervals)  # momentum, volume EMA
        get_CMO(df, col_name, intervals)  # momentum
        get_SMA(df, col_name, intervals)
        get_SMA(df, 'open', intervals)
        get_EMA(df, col_name, intervals)
        get_WMA(df, col_name, intervals)
        get_HMA(df, col_name, intervals)
        get_TRIX(df, col_name, intervals)  # trend
        get_CCI(df, col_name, intervals)  # trend
        get_DPO(df, col_name, intervals)  # Trend oscillator
        get_kst(df, col_name, intervals)  # Trend
        get_DMI(df, col_name, intervals)  # trend
        get_BB_MAV(df, col_name, intervals)  # volatility
        # get_PSI(df, col_name, intervals)  # can't find formula
        get_force_index(df, intervals)  # volume
        get_kdjk_rsv(df, intervals)  # ready to use, +2*len(intervals), 2 rows
        get_EOM(df, col_name, intervals)  # volume momentum
        get_volume_delta(df)  # volume +1
        get_IBR(df)  # ready to use +1

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

                # min_ = np.inf
                # min_index = -1
                # max_ = -np.inf
                # max_index = -1
                # for i in range(window_begin, window_end + 1):
                #     price = df.iloc[i][col_name]
                #     if price < min_:
                #         min_ = price
                #         min_index = i
                #     if price > max_:
                #         max_ = price
                #         max_index = i

                high_window = df.loc[window_begin : window_end]['high']
                low_window = df.loc[window_begin : window_end]['low']

                max_index = high_window.idxmax()
                min_index = low_window.idxmin()
                max_ = high_window.max()
                min_ = low_window.min()

                min_after = df.loc[max_index : max_index+12]['low'].min()
                max_after = df.loc[max_index : max_index+12]['high'].max()

                if max_index == window_middle and max_ - min_after > 5:
                    labels[window_middle] = 0
                elif min_index == window_middle and max_after - min_ > 5:
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
            df = self.get_trend(df, technical_info)

            intervals = range(6, 27)  # 21
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

            prev_len = len(df)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(self.output_path, index=False)
        else:
            print("labels already calculated")
        return df

    def feature_selection(self):
        df_batch = self.df_by_date(None, 10)
        list_features = list(df_batch.loc[:, self.start_col:self.end_col].columns)
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        x_train = mm_scaler.fit_transform(df_batch.loc[:, self.start_col:self.end_col].values)
        y_train = df_batch['labels'].values
        num_features = 256  # should be a perfect square
        topk = 350
        select_k_best = SelectKBest(f_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        common = list(set(selected_features_anova).intersection(selected_features_mic))
        if len(common) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topK"'.format(len(common),
                                                                                                    num_features))
        feat_idx = []
        for c in common:
            feat_idx.append(list_features.index(c))
        feat_idx = sorted(feat_idx[0:225])
        return feat_idx

    def df_by_date(self, start_date=None, years=5):
        if not start_date:
            start_date = self.df.head(1).iloc[0]["timestamp"]

        end_date = start_date + pd.offsets.DateOffset(years=years)
        df_batch = self.df[(self.df["timestamp"] >= start_date) & (self.df["timestamp"] <= end_date)]
        return df_batch

    def get_data(self, start_date=None, years=5):
        df_batch = self.df_by_date(start_date, years)
        x = df_batch.loc[:, self.start_col:self.end_col].values
        x = x[:, self.feat_idx]
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        x = mm_scaler.fit_transform(x)
        dim = int(np.sqrt(x.shape[1]))
        x = reshape_as_image(x, dim, dim)
        x = np.stack((x,) * 3, axis=-1)

        y = df_batch['labels'].values
        sample_weights = self.get_sample_weights(y)
        y = self.one_hot_enc.transform(y.reshape(-1, 1))

        return x, y, df_batch, sample_weights

    def get_sample_weights(self, y):
        """
        calculate the sample weights based on class weights. Used for models with
        imbalanced data and one hot encoding prediction.

        params:
            y: class labels as integers
        """

        y = y.astype(int)  # compute_class_weight needs int labels
        class_weights = compute_class_weight('balanced', np.unique(y), y)

        print("real class weights are {}".format(class_weights), np.unique(y))
        print("value_counts", np.unique(y, return_counts=True))
        sample_weights = y.copy().astype(float)
        for i in np.unique(y):
            sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
            # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

        return sample_weights

    def get_rolling_data_next(self, start_date=None, window_size_yrs=6, cross_val_split=0.2):
        if not start_date:
            start_date = self.batch_start_date

        x_train, y_train, df_batch_train, sample_weights = self.get_data(start_date, window_size_yrs)
        train_end_date = df_batch_train.tail(1).iloc[0]["timestamp"]
        test_start_date = train_end_date + pd.offsets.DateOffset(days=1)
        test_end_date = test_start_date + pd.offsets.DateOffset(years=self.test_duration_years)
        x_test, y_test, df_batch_test, _ = self.get_data(test_start_date, self.test_duration_years)
        x_train, x_cv, y_train, y_cv, sample_weights, _ = train_test_split(x_train, y_train, sample_weights,
                                                                           train_size=1 - cross_val_split,
                                                                           test_size=cross_val_split,
                                                                           random_state=2, shuffle=True,
                                                                           stratify=y_train)

        self.batch_start_date = self.batch_start_date + pd.offsets.DateOffset(years=1)
        is_last_batch = False
        if (self.df.tail(1).iloc[0]["timestamp"] - test_end_date).days < 180:  # 6 months
            is_last_batch = True
        return x_train, y_train, x_cv, y_cv, x_test, y_test, df_batch_train, df_batch_test, \
               sample_weights, is_last_batch
