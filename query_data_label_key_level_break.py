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

START_TIME_M15 = "2016-01-01 00:00:00"
END_TIME_M15 = "2020-01-01 00:00:00"

class EAData:
    def __init__(self, database: str, start_time: str, end_time: str) -> None:
        self.database = database
        self.start_time = start_time
        self.end_time = end_time
        # Create a MySQL database connection with pymysql
        self.db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
        self.engine = create_engine(self.db_url, echo=False)
        self.timeframe = "table_h1"

    def get_data(self) -> str:
        """Get data from database

        Returns:
            string: str: SQL Query
        """
        logger.info(f"Database URL: {self.db_url}")
        sql_query = f"SELECT id, date_time as Date, Open as open, High as high, Low as low, Close as close, Volume as volume, indicator_data, output_ta FROM {self.database} WHERE date_time BETWEEN '{self.start_time}'  AND '{self.end_time}' order by date_time"
        logger.info(f"SQL Query: {sql_query}")
        return sql_query
    
    def caculate(self) -> None:
        """Calculate data from database
        - Chart Pattern
        - Save data to Database and backup to csv
        """
        sql_query = self.get_data()
        df_pd = pd.read_sql(sql_query, con=self.engine)
        df_pd["labels"] = 2  # Mặc định nhãn là 2 (không đủ điều kiện)
        number_logic_true = 0
        list_break_resistance_label = []
        list_break_support_label = []

        list_break_resistance = []
        list_break_support = []
        
        # Parse indicator_data và thêm các cột tương ứng
        for index, row in tqdm.tqdm(df_pd.iterrows()):
            indicator_data = eval(row["indicator_data"].replace('NaN', '2').replace('Infinity', '0').replace('false', '0').replace('true', '1'))
            for key, value in indicator_data.items():
                df_pd.loc[index, key] = value 
            if index > 5: 
                window_close = df_pd['close'].iloc[index-5:index+1]
                status_max = window_close.idxmax() == index
                status_min = window_close.idxmin() == index
                    
                timestamp = df_pd.loc[index, 'timestamp']
                if timestamp == '2021-02-01 02:00:00':
                    print(timestamp)

                if df_pd.loc[index, 'output_ta'] == "":
                    continue
                output_ta = json.loads(df_pd.loc[index, 'output_ta'])
                key_level = output_ta['key_level_v0']

                if len(key_level['support1']) > 0:
                    current_support = key_level['support1'][0]
                else:
                    current_support = 0
                
                if len(key_level['resistance1']) > 0:
                    current_resistance = key_level['resistance1'][1]
                else:
                    current_resistance = 0
                

                # #   Tính lại giá trị thực của key levels
                current_delta_support1 = df_pd.loc[index, 'support1']
                current_delta_resistance1 = df_pd.loc[index, 'resistance1']

                # # Deltas quá khứ
                last_delta_support1 = df_pd.loc[index - 1, 'support1'] if index > 0 else 0
                last_delta_resistance1 = df_pd.loc[index - 1, 'resistance1'] if index > 0 else 0
                
                rsi = df_pd.loc[index-1, 'rsi_14']
                delta_threshold_1 = 5
                TP_price = 20
                SL_price = 15

                # Thống kê 
                # if current_delta_resistance1 < -delta_threshold_1 and last_delta_resistance1 > -delta_threshold_1:
                #     list_break_resistance.append((timestamp, current_resistance, "R"))
                #     for j in range(index+1, min(index+100, len(df_pd))):
                #         if df_pd.loc[index, 'close'] - df_pd.loc[j, 'low'] >= SL_price:
                #             break
                #         elif df_pd.loc[j, 'high'] - df_pd.loc[index, 'close'] >= TP_price:
                #             number_logic_true += 1
                #             break
                        
                # if current_delta_support1 > delta_threshold_1 and last_delta_support1 < delta_threshold_1:
                #     list_break_support.append((timestamp, current_support, "S"))
                #     for j in range(index+1, min(index+100, len(df_pd))):
                #         if df_pd.loc[j, 'high'] - df_pd.loc[index, 'close'] >= SL_price:
                #             break
                #         elif df_pd.loc[index, 'close'] - df_pd.loc[j, 'low'] >= TP_price:
                #             number_logic_true += 1
                #             break
                
                # === BUY Break conditions ===
                if (
                    (current_delta_resistance1 < -2 and status_max and last_delta_resistance1 > -2 ) and rsi > 50
                ):
                    current_resistance = df_pd.loc[index, 'resistance1'] + df_pd.loc[index, 'close']
                    for j in range(index, min(index+6, len(df_pd))):
                        if current_resistance - df_pd.loc[j, 'close'] < -8:
                            df_pd.loc[j, 'labels'] = 1
                            list_break_resistance_label.append((timestamp, current_resistance, "R"))
                            break
                        
                    # for j in range(index, min(index+50, len(df_pd))):
                    #     if df_pd.loc[index-1, 'close'] - df_pd.loc[j, 'low'] >= SL_price:
                    #         break
                    #     elif df_pd.loc[j, 'high'] - df_pd.loc[index-1, 'close'] >= TP_price:
                    #         number_logic_true += 1
                    #         break
                # === SELL Break conditions ===
                if (
                    (current_delta_support1 > 2 and status_min and last_delta_support1 < 2) and rsi < 50
                ):
                    current_support = df_pd.loc[index, 'support1'] + df_pd.loc[index, 'close']
                    for j in range(index, min(index+6, len(df_pd))):
                        if current_support - df_pd.loc[j, 'close'] > 8:
                            df_pd.loc[j, 'labels'] = 0
                            list_break_support_label.append((timestamp, current_support, "S"))
                            break
                    # for j in range(index, min(index+50, len(df_pd))):
                    #     if df_pd.loc[j, 'high'] - df_pd.loc[index-1, 'close'] >= SL_price:
                    #         break
                    #     elif df_pd.loc[index-1, 'close'] - df_pd.loc[j, 'low'] >= TP_price:
                    #         number_logic_true += 1
                    #         break
            
            print(df_pd["labels"].value_counts())    
        # remove column indicator_data
        df_pd.drop(columns=["indicator_data", "output_ta"], inplace=True)
        df_pd.drop(columns=["id"], inplace=True)
        df_pd.to_csv(f"indicator_data_xau_{self.timeframe}_{START_TIME_M15.split('-')[0]}.csv", index=False)
        print(f"Count logic true: {number_logic_true}")
        print(f"Total KL: {len(list_break_resistance)+len(list_break_support)}")

        # df_break_resistance = pd.DataFrame(list_break_resistance_label, columns=["timestamp", "price", "type"])
        # df_break_support = pd.DataFrame(list_break_support_label, columns=["timestamp", "price", "type"])
        # df_break_resistance.to_csv(f"break_resistance_{self.timeframe}_{START_TIME_M15.split('-')[0]}.csv", index=False)
        # df_break_support.to_csv(f"break_support_{self.timeframe}_{START_TIME_M15.split('-')[0]}.csv", index=False)
        
if __name__ == "__main__":
        chart_pattern = EAData(
            database="exness_xau_usd_h1",   
            start_time=START_TIME_M15,
            end_time=END_TIME_M15,
        )
        chart_pattern.caculate()