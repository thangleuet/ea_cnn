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


DB_HOST = "42.96.41.209"
DB_USER = "xttrade"

DB_PASSWORD ="Xttrade1234$"
DB_NAME = "XTTRADE"

START_TIME_M15 = "2024-01-01 00:00:00"
END_TIME_M15 = "2024-10-10 00:00:00"

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
        sql_query = f"SELECT id, date_time as Date, Open, High, Low, Close, indicator_data, technical_info FROM {self.database} WHERE date_time BETWEEN '{self.start_time}'  AND '{self.end_time}' order by date_time"
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
        price = 15
        df_pd["labels"] = 2
        current_time = None
        
        df_pd["labels"] = 2  # Mặc định là không hành động
        window = 20  # Cửa sổ nhìn lại
        min_move = 10  # Biên độ giá tối thiểu để xác nhận đỉnh/đáy

        # Để lại 10 nến sau để xác nhận
        for index, row in tqdm.tqdm(df_pd.iterrows()): 
            # Để lại 10 nến sau để xác nhận
            if index < window - 1 or index > len(df_pd) - 10:
                continue
            indicator_data = eval(df_pd["indicator_data"].iloc[index].replace('NaN', '2'))
            for key, value in indicator_data.items():
                df_pd.loc[index, key] = value
            # Tính vùng giá tương đối
            past_prices = df_pd["close"].iloc[index - window:index]
            current_price = df_pd["close"].iloc[index]
            rsi_current = df_pd["rsi_14"].iloc[index]
            ema_diff = df_pd["ema_34"].iloc[index] - df_pd["ema_89"].iloc[index]
            ema_short_diff = df_pd["ema_14"].iloc[index] - df_pd["ema_34"].iloc[index]
            
            # Tính RSI trung bình và STD
            rsi_mean = df_pd["rsi_14"].iloc[index - window:index].mean()
            rsi_std = df_pd["rsi_14"].iloc[index - window:index].std()
            
            # Label 0: Bán tại đỉnh
            if (current_price >= np.percentile(past_prices, 90) and 
                rsi_current > max(70, rsi_mean + rsi_std) and 
                ema_short_diff < 0):  # EMA ngắn hạn đảo chiều xuống
                future_prices = df_pd["close"].iloc[index:index + 10]
                if min(future_prices) < current_price - min_move:
                    df_pd.loc[index, "labels"] = 0
            
            # Label 1: Mua tại đáy
            elif (current_price <= np.percentile(past_prices, 10) and 
                rsi_current < min(30, rsi_mean - rsi_std) and 
                ema_short_diff > 0):  # EMA ngắn hạn đảo chiều lên
                future_prices = df_pd["close"].iloc[index:index + 10]
                if max(future_prices) > current_price + min_move:
                    df_pd.loc[index, "labels"] = 1  
            # count label
            print(df_pd["labels"].value_counts())    
        # remove column indicator_data
        df_pd.drop(columns=["id", "indicator_data","technical_info"], inplace=True)
        df_pd.to_csv(f"indicator_data_xau_{self.timeframe}_{START_TIME_M15.split('-')[0]}_{price}.csv", index=False)
            
if __name__ == "__main__":
        chart_pattern = EAData(
            database="exness_xau_usd_m15",
            start_time=START_TIME_M15,
            end_time=END_TIME_M15,
        )
        chart_pattern.caculate()