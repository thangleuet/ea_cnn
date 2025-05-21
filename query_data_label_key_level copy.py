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

START_TIME_M15 = "2022-01-01 00:00:00"
END_TIME_M15 = "2023-01-01 00:00:00"

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
        """Calculate data from database
        - Chart Pattern
        - Save data to Database and backup to csv
        """
        sql_query = self.get_data()
        df_pd = pd.read_sql(sql_query, con=self.engine)
        df_pd["labels"] = 2  # Mặc định nhãn là 2 (không đủ điều kiện)

        # Parse indicator_data và thêm các cột tương ứng
        for index, row in tqdm.tqdm(df_pd.iterrows()):
            if index > 5:
                indicator_data = eval(row["indicator_data"].replace('NaN', '2').replace('Infinity', '0').replace('false', '0').replace('true', '1'))
                for key, value in indicator_data.items():
                    df_pd.loc[index, key] = value  
                    
                window_close = df_pd['close'].iloc[index-5:index+1]
                status_max = window_close.idxmax() == index - 1
                status_min = window_close.idxmin() == index - 1
                    
                timestamp = df_pd.loc[index, 'timestamp']
                if timestamp == '2022-02-08 16:00:00':
                    print(timestamp)
                if status_max or status_min:
                    timestamp = 1
                #   Tính lại giá trị thực của key levels
                current_support1 = df_pd.loc[index, 'support1'] + df_pd.loc[index, 'close']
                current_support2 = df_pd.loc[index, 'support2'] + df_pd.loc[index, 'close']
                current_resistance1 = df_pd.loc[index, 'resistance1'] + df_pd.loc[index, 'close']
                current_resistance2 = df_pd.loc[index, 'resistance2'] + df_pd.loc[index, 'close']

                # Deltas quá khứ
                last_support1 = df_pd.loc[index - 1, 'support1'] + df_pd.loc[index - 1, 'close']
                last_support2 = df_pd.loc[index - 1, 'support2'] + df_pd.loc[index - 1, 'close']
                last_resistance1 = df_pd.loc[index - 1, 'resistance1'] + df_pd.loc[index - 1, 'close']
                last_resistance2 = df_pd.loc[index - 1, 'resistance2'] + df_pd.loc[index - 1, 'close']
                
                diff_ema_5 = df_pd.loc[index, 'close'] - df_pd.loc[index, 'ema_5']
                candle_type = df_pd.loc[index, 'candle_type']
                atr = df_pd.loc[index, 'atr']
                delta_threshold_1 = 5
                TP_price = 2 * atr
                SL_price = atr
                
                if status_min:
                    current_delta_support_1 = current_support1 - df_pd.loc[index, 'low']
                    current_delta_support_2 = current_support2 - df_pd.loc[index, 'low']
                    last_delta_support_1 = last_support1 - df_pd.loc[index, 'low']
                    last_delta_support_2 = last_support2 - df_pd.loc[index, 'low']
                    if abs(current_delta_support_1) < 3 or abs(current_delta_support_2) < 3 or abs(last_delta_support_1) < 3 or abs(last_delta_support_2) < 3:
                        df_pd.loc[index, 'labels'] = 1
                    elif current_delta_support_1 > 3 and current_support1 - df_pd.loc[index, 'close'] < 3:
                        df_pd.loc[index, 'labels'] = 1
                    elif current_delta_support_2 > 3 and current_support2 - df_pd.loc[index, 'close'] < 3:
                        df_pd.loc[index, 'labels'] = 1
                    elif last_delta_support_1 > 3 and last_support1 - df_pd.loc[index, 'close'] < 3:
                        df_pd.loc[index, 'labels'] = 1
                    elif last_delta_support_2 > 3 and last_support2 - df_pd.loc[index, 'close'] < 3:
                        df_pd.loc[index, 'labels'] = 1
                elif status_max:
                    current_delta_resistance_1 = current_resistance1 - df_pd.loc[index, 'high']
                    current_delta_resistance_2 = current_resistance2 - df_pd.loc[index, 'high']
                    last_delta_resistance_1 = last_resistance1 - df_pd.loc[index, 'high']
                    last_delta_resistance_2 = last_resistance2 - df_pd.loc[index, 'high']
                    if abs(current_delta_resistance_1) < 3 or abs(current_delta_resistance_2) < 3 or abs(last_delta_resistance_1) < 3 or abs(last_delta_resistance_2) < 3:
                        df_pd.loc[index, 'labels'] = 0
                    elif current_delta_resistance_1 < -3 and current_resistance1 - df_pd.loc[index, 'close'] > -3:
                        df_pd.loc[index, 'labels'] = 0
                    elif current_delta_resistance_2 < -3 and current_resistance2 - df_pd.loc[index, 'close'] > -3:
                        df_pd.loc[index, 'labels'] = 0
                    elif last_delta_resistance_1 < -3 and last_resistance1 - df_pd.loc[index, 'close'] > -3:
                        df_pd.loc[index, 'labels'] = 0
                    elif last_delta_resistance_2 < -3 and last_resistance2 - df_pd.loc[index, 'close'] > -3:
                        df_pd.loc[index, 'labels'] = 0
                    
                
                # if (
                #     (-delta_threshold_1 < last_support1 < delta_threshold_1 and last_support1 > current_support1 and status_min)
                #     or (-delta_threshold_1 < last_support2 < delta_threshold_1 and last_support2 > current_support2 and status_min)
                #     # or (current_delta_resistance1 < -3 and last_delta_resistance1 > -3)
                #     # or (current_delta_resistance2 < -3 and last_delta_resistance2 > -3)
                # ):
                #     # df_pd.loc[index, 'labels'] = 1
                #     for j in range(index+1, min(index+50, len(df_pd))):
                #         if df_pd.loc[index, 'close'] - df_pd.loc[j, 'low'] >= SL_price:
                #             break
                #         elif df_pd.loc[j, 'high'] - df_pd.loc[index, 'close'] >= TP_price:
                #             df_pd.loc[index, 'labels'] = 1
                #             break
                # # === SELL conditions ===
                # if (
                #     (delta_threshold_1 > last_resistance1 > -delta_threshold_1 and last_resistance1 < current_resistance1 and status_max)
                #     or (delta_threshold_1 > last_resistance2 > -delta_threshold_1 and last_resistance2 < current_resistance2 and status_max)
                #     # or (current_delta_support1 > 3 and last_delta_support1 < 3)
                #     # or (current_delta_support2 > 3 and last_delta_support2 < 3)
                # ):
                #     # df_pd.loc[index, 'labels'] = 0
                #     for j in range(index+1, min(index+50, len(df_pd))):
                #         if df_pd.loc[j, 'high'] - df_pd.loc[index, 'close'] >= SL_price:
                #             break
                #         elif df_pd.loc[index, 'close'] - df_pd.loc[j, 'low'] >= TP_price:
                #             df_pd.loc[index, 'labels'] = 0
                #             break
            
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