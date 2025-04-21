from datetime import datetime
import json
import os
import time
import numpy as np
from sqlalchemy import create_engine
from loguru import logger
import pandas as pd
import sys
import ast

import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from sqlalchemy import text
import sqlalchemy


DB_HOST = "42.96.41.209"
DB_USER = "xttrade"

DB_PASSWORD ="Xttrade1234$"
DB_NAME = "XTTRADE"

START_TIME_M15 = "2021-01-01 00:00:00"
END_TIME_M15 = "2024-10-10 00:00:00"

CANDLE_PATTERN = [1, 2, 3, 4, 6, 7, 8, 9]
CANDLE_DOUBLE = [10, 11, 12, 13, 14, 15]
CANDLE_TRIPPLE_PATTERN = [16, 17, 18, 19, 20, 21]

class EAData:
    def __init__(self, database: str, start_time: str, end_time: str) -> None:
        self.database = database
        self.start_time = start_time
        self.end_time = end_time
        # Create a MySQL database connection with pymysql
        self.db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
        self.engine = create_engine(self.db_url, echo=False)
        self.timeframe = "table_m15"

    def get_data(self) -> str:
        """Get data from database

        Returns:
            string: str: SQL Query
        """
        logger.info(f"Database URL: {self.db_url}")
        sql_query = f"SELECT id, Open, High, Low, Close, indicator_data FROM {self.database} WHERE date_time BETWEEN '{self.start_time}'  AND '{self.end_time}' order by date_time"
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
    
    def update_label_to_database(self, table_name, date_time, type_order, type_trend, connection):
        try:
            upsert_query = """
            INSERT INTO data_label_ai (date_time, label, type_trend, table_name)
            VALUES (:date_time, :label, :type_trend, :table_name)
            ON DUPLICATE KEY UPDATE 
                label = VALUES(label),
                type_trend = VALUES(type_trend),
                table_name = VALUES(table_name)
            """
            params = {
                "date_time": pd.to_datetime(date_time).strftime('%Y-%m-%d %H:%M:%S'),
                "label": type_order,
                "type_trend": type_trend,
                "table_name": table_name
            }
            
            connection.execute(sqlalchemy.text(upsert_query), params)
            connection.commit()
        except Exception as e:
            logger.error(f"Error updating label to database: {e}")
    
    def caculate(self) -> None:
        """Caculate data from database
        - Chart Pattern
        - Save data to Database and backup to csv
        """
        sql_query = self.get_data()
        df_pd = pd.read_sql(sql_query, con=self.engine)
        window_size = 15
        price = 10
        df_pd["labels"] = 2
        lookback_period = 50  # Có thể điều chỉnh số lượng nến nhìn lại
        # save to csv
        with self.engine.connect() as connection:
            for index, row in tqdm.tqdm(df_pd.iterrows()):
                indicator_data = eval(row["indicator_data"].replace('NaN', '2'))
                for key, value in indicator_data.items():
                    df_pd.loc[index, key] = value

            # for index, row in tqdm.tqdm(df_pd.iterrows()):        
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
                    current_price = df_pd.loc[window_middle, 'close']
                    diff_ema_34_89 = df_pd.loc[window_middle, 'ema_34'] - df_pd.loc[window_middle, 'ema_89']
                    
                    type_trend = ''

                    # Lấy giá trị cao/thấp từ từng chỉ báo trong cửa sổ
                    for indicator in indicator_window:
                        indicator = eval(indicator.replace('NaN', '2'))
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
                            indicator = eval(indicator.replace('NaN', '2'))
                            high = indicator["high"]
                            low = indicator["low"]
                            high_values_after.append(high)
                            low_values_after.append(low)

                        max_index_after, max_after = self.find_first_greater(high_values_after, max_value, True)
                        if len(low_values_after[:max_index_after]) > 0:
                            min_after = min(low_values_after[:max_index_after])
                        else:
                            min_after = max_value
                            
                    elif min_index == window_middle:
                        indicator_window_after = df_pd.loc[min_index + 1 : min(window_end+100, len(df_pd))]["indicator_data"]
                        for indicator in indicator_window_after:
                            indicator = eval(indicator.replace('NaN', '2'))
                            high = indicator["high"]
                            low = indicator["low"]
                            high_values_after.append(high)
                            low_values_after.append(low)

                        min_index_after, min_after = self.find_first_greater(low_values_after, min_value, False)
                        if len(high_values_after[:min_index_after]) > 0:
                            max_after = max(high_values_after[:min_index_after])
                        else:
                            max_after = min_value
                             
                    # Kiểm tra đỉnh/đáy trước đó (lookback)
                    if df_pd.loc[window_middle, 'count_ema_34_ema_89'] > 150:
                        lookback_period = 100
                    else:
                        lookback_period = 50
                    lookback_start = max(0, window_middle - lookback_period)  # Đảm bảo không vượt quá index 0
                    lookback_data = df_pd.loc[lookback_start:window_middle - 5]  # Dữ liệu trước window_middle
                    previous_high = lookback_data['high'].max() if not lookback_data.empty else current_price
                    previous_high_index = lookback_data['high'].idxmax() if not lookback_data.empty else window_middle
                    min_middle = lookback_data['low'].loc[previous_high_index:window_middle].min()
                    previous_low = lookback_data['low'].min() if not lookback_data.empty else current_price
                    previous_low_index = lookback_data['low'].idxmin() if not lookback_data.empty else window_middle
                    max_middle = lookback_data['high'].loc[previous_high_index:window_middle].max()
                    
                    # Counter trend 1
                    condition_price_high_1 = abs(df_pd.loc[previous_high_index, 'high'] - df_pd.loc[previous_high_index, 'ema_34']) > 5 and abs(df_pd.loc[previous_high_index, 'ema_34'] - df_pd.loc[previous_high_index, 'ema_89']) > 5
                    condition_price_low_1 = abs(df_pd.loc[previous_low_index, 'low'] - df_pd.loc[previous_low_index, 'ema_34']) > 5 and abs(df_pd.loc[previous_low_index, 'ema_34'] - df_pd.loc[previous_low_index, 'ema_89']) > 5
                    # Counter trend big move
                    condition_high_bigmove = (df_pd.loc[previous_high_index, 'high'] - df_pd['low'].loc[previous_high_index-10:previous_high_index].min()) > 20
                    condition_low_bigmove = (df_pd['high'].loc[previous_low_index-10:previous_low_index].max() - df_pd.loc[previous_low_index, 'low']) > 20
                    
                    # Điều kiện confirm đỉnh thấp hơn, đáy cao hơn
                    condition_down = (previous_high - df_pd.loc[window_middle, 'high'] > -2) and (df_pd.loc[previous_high_index, 'ema_34'] - df_pd.loc[previous_high_index, 'ema_89']) > -2 and df_pd.loc[window_middle, 'high'] - min_middle > 3
                    condition_up = (previous_low - df_pd.loc[window_middle, 'low'] < 2) and (df_pd.loc[previous_low_index, 'ema_34'] - df_pd.loc[previous_low_index, 'ema_89']) < 2 and max_middle - df_pd.loc[window_middle, 'low'] > 3
                    
                    if max_index == window_middle and (current_price - min_after) >= price:
                        condition_touch = df_pd.loc[window_middle, 'ema_34'] - 2 < df_pd.loc[window_middle, 'low'] < df_pd.loc[window_middle, 'ema_89'] + 2 
                        for i in range(1, 3):
                            diff_ema_34_89_after = (df_pd.loc[window_middle+i, 'ema_34'] - df_pd.loc[window_middle+i, 'ema_89'])
                            rsi_after = df_pd.loc[window_middle+i, 'rsi_14']
                            diff_ema_89 = (df_pd.loc[window_middle+i, 'close'] - df_pd.loc[window_middle+i-1, 'ema_89'])
                            if (df_pd.loc[window_middle, 'high'] - df_pd.loc[window_middle+i, 'close']) < 3 and (df_pd.loc[window_middle+i, 'close'] - min_after) >= price:
                                    if diff_ema_34_89_after < 0:
                                        df_pd.loc[window_middle+i, 'labels'] = 0  # SELL
                                        type_trend = 'Follow Trend'
                                    elif diff_ema_34_89_after > 0 and condition_down and diff_ema_89 > 5:
                                        df_pd.loc[window_middle+i, 'labels'] = 0 # SELL
                                        if condition_price_high_1:
                                            type_trend = "Counter Trend"
                                        elif condition_high_bigmove:
                                            type_trend = "Counter Trend Big Move"
                    elif min_index == window_middle and (max_after - current_price) >= price:
                        condition_touch = df_pd.loc[window_middle, 'ema_89'] - 2 < df_pd.loc[window_middle, 'high'] < df_pd.loc[window_middle, 'ema_34'] + 2 
                        # if 3 > df_pd.loc[window_middle, 'close'] - df_pd.loc[window_middle, 'open'] > 0:
                        #     if diff_ema_34_89 > 0 and condition_touch:
                        #         df_pd.loc[window_middle, 'labels'] = 1   # BUY 
                        #     elif diff_ema_34_89 < 0 and condition_up:
                        #         df_pd.loc[window_middle, 'labels'] = 1  # BUY
                        for i in range(1, 3):
                            diff_ema_34_89_after = (df_pd.loc[window_middle, 'ema_34'] - df_pd.loc[window_middle, 'ema_89'])
                            rsi_after = df_pd.loc[window_middle+i, 'rsi_14']
                            diff_ema_89 = (df_pd.loc[window_middle+i, 'close'] - df_pd.loc[window_middle+i-1, 'ema_89'])
                            if (df_pd.loc[window_middle + i, 'close'] - df_pd.loc[window_middle, 'low']) < 3 and (max_after - df_pd.loc[window_middle + i, 'close']) >= price:
                                    if diff_ema_34_89_after > 0:
                                        df_pd.loc[window_middle+i, 'labels'] = 1 # BUY
                                        type_trend = 'Follow Trend'
                                    elif diff_ema_34_89_after < 0 and condition_up and diff_ema_89 < -5:
                                        df_pd.loc[window_middle+i, 'labels'] = 1 # BUY
                                        if condition_price_low_1:
                                            type_trend = 'Counter Trend'
                                        elif condition_low_bigmove:
                                            type_trend = 'Counter Trend Big Move'
                                    
                    for i in range(1, 3):                   
                        type_order = df_pd.loc[window_middle+i, 'labels']
                        date_time = df_pd.loc[window_middle+i, 'timestamp']
                        table_name = self.database
                        self.update_label_to_database(table_name, date_time, type_order, type_trend, connection)
                    # count label
                    print(df_pd["labels"].value_counts())    
        # remove column indicator_data
        df_pd.drop(columns=["indicator_data"], inplace=True)
        df_pd.drop(columns=["id"], inplace=True)
        df_pd.to_csv(f"indicator_data_xau_{self.timeframe}_{START_TIME_M15.split('-')[0]}_{price}.csv", index=False)
            
if __name__ == "__main__":
        chart_pattern = EAData(
            database="exness_xau_usd_m15",
            start_time=START_TIME_M15,
            end_time=END_TIME_M15,
        )
        chart_pattern.caculate()