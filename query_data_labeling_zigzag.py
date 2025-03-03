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
        
        # save to csv
        for index, row in tqdm.tqdm(df_pd.iterrows()):
            indicator_data = eval(row["indicator_data"].replace('NaN', '2'))
            for key, value in indicator_data.items():
                df_pd.loc[index, key] = value

            # labeling data
            technical_info = eval(row["technical_info"].replace('NaN', '2'))
            x_zigzag_data = technical_info["zigzag_data"]["x_zigzag_data"][:-3]
            y_zigzag_data = technical_info["zigzag_data"]["y_zigzag_data"][:-3]
            type_zigzag_data = technical_info["zigzag_data"]["type_zigzag_data"][:-3]
            
            rsi = df_pd.loc[index, 'rsi_14']
            
            for i, (x, y, t) in enumerate(zip(x_zigzag_data, y_zigzag_data, type_zigzag_data)):
                if current_time is not None and pd.to_datetime(x) <= pd.to_datetime(current_time):
                    continue
                if t == "high" and rsi > 60:
                    for j in range(i + 1, len(y_zigzag_data)):
                        if type_zigzag_data[j] == "low" and (y - y_zigzag_data[j] > price) and (pd.to_datetime(x_zigzag_data[j]) - pd.to_datetime(x)).total_seconds() > 15 * 5 * 60:
                            if all(y_zigzag_data[k] < y for k in range(i + 1, j+1)):
                                index_entry = df_pd.index[df_pd["Date"] == x]
                                if index_entry.empty:
                                    break
                                diff_ema = df_pd.loc[index_entry[0], 'ema_34'] - df_pd.loc[index_entry[0], 'ema_89']
                                if diff_ema < 0:
                                    df_pd.loc[index_entry[0], "labels"] = 0
                                    current_time = x
                                elif diff_ema > 0 and df_pd.loc[index_entry[0], 'close'] - y_zigzag_data[j] > 20:
                                    df_pd.loc[index_entry[0], "labels"] = 0
                                    current_time = x
                                
                                for i in range(1, 2): 
                                    index_entry_after = df_pd.index[df_pd["Date"] == x]+i
                                    if index_entry_after.empty:
                                        break
                                    diff_price = abs(df_pd.loc[index_entry[0], 'close'] - df_pd.loc[index_entry_after[0], 'close'])
                                    if diff_price < 3 and df_pd.loc[index_entry_after[0], 'rsi_14'] > 60:
                                        diff_ema = df_pd.loc[index_entry_after[0], 'ema_34'] - df_pd.loc[index_entry_after[0], 'ema_89']
                                        if diff_ema < 0:
                                            df_pd.loc[index_entry_after, "labels"] = 0
                                            current_time = x
                                        elif diff_ema > 0 and df_pd.loc[index_entry_after[0], 'close'] - y_zigzag_data[j] > 20:
                                            df_pd.loc[index_entry_after, "labels"] = 0
                                            current_time = x
                            break  

                elif t == "low" and rsi < 40:
                    for j in range(i + 1, len(y_zigzag_data)):
                        if type_zigzag_data[j] == "high" and (y_zigzag_data[j] - y > price) and (pd.to_datetime(x_zigzag_data[j]) - pd.to_datetime(x)).total_seconds() > 15 * 5 * 60:
                            if all(y_zigzag_data[k] > y for k in range(i + 1, j+1)):
                                index_entry = df_pd.index[df_pd["Date"] == x]
                                if index_entry.empty:
                                    break
                                diff_ema = df_pd.loc[index_entry[0], 'ema_34'] - df_pd.loc[index_entry[0], 'ema_89']
                                if diff_ema > 0:
                                    df_pd.loc[index_entry[0], "labels"] = 1
                                    current_time = x
                                elif diff_ema < 0 and df_pd.loc[index_entry[0], 'close'] - y_zigzag_data[j] > 20:
                                    df_pd.loc[index_entry[0], "labels"] = 1
                                    current_time = x

                                for i in range(1, 2): 
                                    index_entry_after = df_pd.index[df_pd["Date"] == x]+i
                                    if index_entry_after.empty:
                                        break
                                    diff_price = abs(df_pd.loc[index_entry_after[0], 'close'] - df_pd.loc[index_entry[0], 'close'])
                                    if diff_price < 3 and df_pd.loc[index_entry_after[0], 'rsi_14'] < 40:
                                        diff_ema = df_pd.loc[index_entry_after[0], 'ema_34'] - df_pd.loc[index_entry_after[0], 'ema_89']
                                        if diff_ema > 0:
                                            df_pd.loc[index_entry_after, "labels"] = 1
                                            current_time = x
                                        elif diff_ema < 0 and df_pd.loc[index_entry_after[0], 'close'] - y_zigzag_data[j] > 20:
                                            df_pd.loc[index_entry_after, "labels"] = 1
                                            current_time = x
                            break  
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