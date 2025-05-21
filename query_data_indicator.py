from datetime import datetime
import json
import os
import time
import numpy as np
from sqlalchemy import create_engine
from loguru import logger
import pandas as pd
import sys

import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from sqlalchemy import text


DB_HOST = "42.96.41.209"
DB_USER = "xttrade"

DB_PASSWORD ="Xttrade1234$"
DB_NAME = "XTTRADE"

START_TIME_M15 = "2023-01-01 00:00:00"
END_TIME_M15 = "2025-01-01 00:00:00"

class EAData:
    def __init__(self, database: str, start_time: str, end_time: str) -> None:
        self.database = database
        self.start_time = start_time
        self.end_time = end_time
        # Create a MySQL database connection with pymysql
        self.db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
        self.engine = create_engine(self.db_url, echo=False)
        self.timeframe = "table_m5"

    def get_data(self) -> str:
        """Get data from database

        Returns:
            string: str: SQL Query
        """
        logger.info(f"Database URL: {self.db_url}")
        sql_query = f"SELECT id, date_time as Date, Open as open, High as high, Low as low, Close as close, Volume as volume, indicator_data FROM {self.database} WHERE date_time BETWEEN '{self.start_time}'  AND '{self.end_time}' order by date_time"
        logger.info(f"SQL Query: {sql_query}")
        return sql_query
    
    def get_nearest_larger_index(self, n, df):
        # Lấy danh sách các index thỏa mãn điều kiện df.loc[index, "count_ema_34"] == 0
        indices = df.index[df["count_ema_34"] == 0].tolist()
        
        # Lọc ra các index lớn hơn n
        larger_indices = [idx for idx in indices if idx > n]
        
        # Trả về phần tử gần nhất lớn hơn n (nếu có)
        return min(larger_indices) if larger_indices else None
    
    def find_first_greater(self, values, x, status):
        for index, value in enumerate(values):
            if status:
                if value > x:
                    return index, value
            else:
                if value < x:
                    return index, value
        return len(values)-1, values[-1]  # Nếu không có phần tử nào lớn hơn x
    
    def caculate(self) -> None:
        """Caculate data from database
        - Chart Pattern
        - Save data to Database and backup to csv
        """
        sql_query = self.get_data()
        df_pd = pd.read_sql(sql_query, con=self.engine)
        window_size = 9
        # price = 10
        df_pd["labels"] = 2
        k_atr = 1.5
        # save to csv
        for index, row in tqdm.tqdm(df_pd.iterrows()):
            indicator_data = eval(row["indicator_data"].replace('NaN', '2').replace('Infinity', '0').replace('false', '0').replace('true', '1'))
            for key, value in indicator_data.items():
                df_pd.loc[index, key] = value      
            # labeling data
            if index >= window_size - 1 and index < len(df_pd):
                window_begin = index - (window_size - 1)
                window_end = index
                window_middle = int((window_begin + window_end) / 2)
                indicator_window = df_pd.loc[window_begin : window_end]["indicator_data"]
                high_values = []
                low_values = []
                high_values_after = []
                low_values_after = []
                close_price = df_pd.loc[window_middle, 'close']
                open_price = df_pd.loc[window_middle, 'open']
                diff_ema_5 = df_pd.loc[window_middle, 'diff_ema_5']
                
                support1 = df_pd.loc[window_middle, 'support1'] if abs(df_pd.loc[window_middle, 'support1']) < abs(df_pd.loc[window_middle-1, 'support1']) else df_pd.loc[window_middle-1, 'support1']
                support2 = df_pd.loc[window_middle, 'support2'] if abs(df_pd.loc[window_middle, 'support2']) < abs(df_pd.loc[window_middle-1, 'support2']) else df_pd.loc[window_middle-1, 'support2']
                resistance1 = df_pd.loc[window_middle, 'resistance1'] if abs(df_pd.loc[window_middle, 'resistance1']) < abs(df_pd.loc[window_middle-1, 'resistance1']) else df_pd.loc[window_middle-1, 'resistance1']
                resistance2 = df_pd.loc[window_middle, 'resistance2'] if abs(df_pd.loc[window_middle, 'resistance2']) < abs(df_pd.loc[window_middle-1, 'resistance2']) else df_pd.loc[window_middle-1, 'resistance2']
                supply1 = df_pd.loc[window_middle, 'supply1'] if abs(df_pd.loc[window_middle, 'supply1']) < abs(df_pd.loc[window_middle-1, 'supply1']) else df_pd.loc[window_middle-1, 'supply1']
                demand1 = df_pd.loc[window_middle, 'demand1'] if abs(df_pd.loc[window_middle, 'demand1']) < abs(df_pd.loc[window_middle-1, 'demand1']) else df_pd.loc[window_middle-1, 'demand1']
                
                atr = df_pd.loc[window_middle, 'atr']
                tp_price = 10  # Ngưỡng giá động
                
                timestamp = df_pd.loc[window_middle, 'timestamp']
                if timestamp == '2024-0-21 12:00:00':
                    print(timestamp)

                # Lấy giá trị cao/thấp từ từng chỉ báo trong cửa sổ
                for indicator in indicator_window:
                    indicator = eval(indicator.replace('NaN', '2').replace('Infinity', '0').replace('false', '0').replace('true', '1'))
                    high = indicator["high"]
                    low = indicator["low"]
                    high_values.append(high)
                    low_values.append(low)
                    
                # Xác định giá cao nhất và thấp nhất trong cửa sổ
                max_index = high_values.index(max(high_values)) + window_begin
                max_value = max(high_values)
                min_index = low_values.index(min(low_values)) + window_begin
                min_value = min(low_values)
                            
                if max_index == window_middle:
                    indicator_window_after = df_pd.loc[max_index + 1 : min(window_end+100, len(df_pd))]["indicator_data"]
                    for indicator in indicator_window_after:
                        indicator = eval(indicator.replace('NaN', '2').replace('Infinity', '0').replace('false', '0').replace('true', '1'))
                        high = indicator["high"]
                        low = indicator["low"]
                        high_values_after.append(high)
                        low_values_after.append(low)

                    max_index_after, max_after = self.find_first_greater(high_values_after, max_value, True)
                    min_after = min(low_values_after[:max_index_after])
                        
                elif min_index == window_middle:
                    indicator_window_after = df_pd.loc[min_index + 1 : min(window_end+100, len(df_pd))]["indicator_data"]
                    for indicator in indicator_window_after:
                        indicator = eval(indicator.replace('NaN', '2').replace('Infinity', '0').replace('false', '0').replace('true', '1'))
                        high = indicator["high"]
                        low = indicator["low"]
                        high_values_after.append(high)
                        low_values_after.append(low)

                    min_index_after, min_after = self.find_first_greater(low_values_after, min_value, False)
                    max_after = max(high_values_after[:min_index_after])
                    
                # if max_index == window_middle and (current_price - min_after) >= tp_price:
                #     for i in range(1, 4):
                #         if (current_price - df_pd.loc[window_middle+i, 'close'])/current_price < 0.0015 and (df_pd.loc[window_middle+i, 'close'] - min_after) >= tp_price:
                #             if diff_ema_34_89 < 0 or (diff_ema_34_89 > 0 and (df_pd.loc[window_middle+i, 'close'] - df_pd.loc[window_middle+i, 'ema_89'])/df_pd.loc[window_middle+i, 'close'] > 0.005):
                #                 df_pd.loc[window_middle+i, 'labels'] = 0 # SELL
                # elif min_index == window_middle and (max_after - current_price) >= tp_price:
                #     for i in range(1, 4):
                #         if (df_pd.loc[window_middle + i, 'close'] - current_price)/current_price < 0.0015 and (max_after - df_pd.loc[window_middle + i, 'close']) >= tp_price:
                #             if diff_ema_34_89 > 0 or (diff_ema_34_89 < 0 and (df_pd.loc[window_middle+i, 'ema_89'] - df_pd.loc[window_middle+i, 'close'])/df_pd.loc[window_middle+i, 'ema_89'] > 0.005):
                #                 df_pd.loc[window_middle+i, 'labels'] = 1 # BUY
                
                if (abs(support1) < 5 and support1 != 0) or (abs(support2) < 5 and support2 != 0) or (abs(supply1) < 5 and supply1 != 0):
                    touch_support = True 
                else:
                    touch_support = False
                
                if (abs(resistance1) < 5 and resistance1 != 0) or (abs(resistance2) < 5 and resistance2 != 0) or (abs(demand1) < 5 and demand1 != 0):
                    touch_resistance = True 
                else:
                    touch_resistance = False
                    
                if support1 < 0  or support2 < 0:
                    up_trend = True
                else:
                    up_trend = False
                if resistance1 > 0 or resistance2 > 0:
                    down_trend = True
                else:
                    down_trend = False
                
                if max_index == window_middle and (close_price - min_after) >= tp_price and (touch_resistance or down_trend):
                    if df_pd.loc[window_middle, 'body_size'] < 0 and diff_ema_5 < 0 and df_pd.loc[window_middle-1, 'diff_ema_5'] > 0:
                        df_pd.loc[window_middle, 'labels'] = 0 # SELL
                    for i in range(1, 4):
                        diff_ema_5_after_second = df_pd.loc[window_middle+i, 'diff_ema_5']
                        diff_ema_5_after_first = df_pd.loc[window_middle+i-1, 'diff_ema_5']
                        if df_pd.loc[window_middle+i, 'close'] - min_after >= tp_price and df_pd.loc[window_middle+i, 'body_size'] < 0 and diff_ema_5_after_second < 0 and diff_ema_5_after_first > 0:
                            df_pd.loc[window_middle+i, 'labels'] = 0 # SELL
                elif min_index == window_middle and (max_after - close_price) >= tp_price and (touch_support or up_trend):
                    if df_pd.loc[window_middle, 'body_size'] > 0 and diff_ema_5 > 0 and df_pd.loc[window_middle-1, 'diff_ema_5'] < 0:
                        df_pd.loc[window_middle, 'labels'] = 1 # BUY
                    for i in range(1, 4):
                        diff_ema_5_after_second = df_pd.loc[window_middle+i, 'diff_ema_5']
                        diff_ema_5_after_first = df_pd.loc[window_middle+i-1, 'diff_ema_5']
                        if (max_after - df_pd.loc[window_middle + i, 'close']) >= tp_price and df_pd.loc[window_middle+i, 'body_size'] > 0 and diff_ema_5_after_second > 0 and diff_ema_5_after_first < 0:
                            df_pd.loc[window_middle+i, 'labels'] = 1 # BUY
                
                # count label
                print(df_pd["labels"].value_counts())    
        # remove column indicator_data
        df_pd.drop(columns=["indicator_data"], inplace=True)
        df_pd.drop(columns=["id"], inplace=True)
        df_pd.to_csv(f"indicator_data_xau_{self.timeframe}_{START_TIME_M15.split('-')[0]}.csv", index=False)
            
if __name__ == "__main__":
        chart_pattern = EAData(
            database="exness_xau_usd_h1",
            start_time=START_TIME_M15,
            end_time=END_TIME_M15,
        )
        chart_pattern.caculate()