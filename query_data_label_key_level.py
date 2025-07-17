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

START_TIME_M15 = "2021-01-01 00:00:00"
END_TIME_M15 = "2022-01-01 00:00:00"

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
        list_support = []
        list_resistance = []
        list_time_support = []
        list_time_resistance = []
        list_touch_support = []
        list_touch_resistance = []
        list_touch_time_support = []
        list_touch_time_resistance = []
        number_revert_true = 0
        support_touch_price = 0
        resistance_touch_price = 0
        support_touch_time = 0
        resistance_touch_time = 0
        number_touch = 0

        index_label_support = 0
        index_label_resistance = 0

        number_logic_true = 0

        # Parse indicator_data và thêm các cột tương ứng
        for index, row in tqdm.tqdm(df_pd.iterrows()):
            indicator_data = eval(row["indicator_data"].replace('NaN', '2').replace('Infinity', '0').replace('false', '0').replace('true', '1'))
            for key, value in indicator_data.items():
                df_pd.loc[index, key] = value 
            if index > 5: 
                window_close = df_pd['close'].iloc[index-5:index+1]
                status_max = window_close.idxmax() == index - 1
                status_min = window_close.idxmin() == index - 1
                    
                timestamp = df_pd.loc[index, 'timestamp']
                if timestamp == '2021-02-01 02:00:00':
                    print(timestamp)

                if df_pd.loc[index, 'output_ta'] == "":
                    continue
                output_ta = json.loads(df_pd.loc[index, 'output_ta'])
                key_level = output_ta['key_level_v0']

                if len(key_level['support1']) > 0:
                    current_support = key_level['support1'][0]
                    if current_support not in list_support:
                        list_support.append(current_support)
                        list_time_support.append(timestamp)
                else:
                    current_support = 0
                
                if len(key_level['resistance1']) > 0:
                    current_resistance = key_level['resistance1'][1]
                    if current_resistance not in list_resistance:
                        list_resistance.append(current_resistance)
                        list_time_resistance.append(timestamp)
                else:
                    current_resistance = 0
                

                # #   Tính lại giá trị thực của key levels
                current_delta_support1 = df_pd.loc[index, 'support1']
                current_delta_support2 = df_pd.loc[index, 'support2']
                current_delta_resistance1 = df_pd.loc[index, 'resistance1']
                current_delta_resistance2 = df_pd.loc[index, 'resistance2']

                # # Deltas quá khứ
                last_delta_support1 = df_pd.loc[index - 1, 'support1'] if index > 0 else 0
                last_delta_support2 = df_pd.loc[index - 1, 'support2'] if index > 0 else 0
                last_delta_resistance1 = df_pd.loc[index - 1, 'resistance1'] if index > 0 else 0
                last_delta_resistance2 = df_pd.loc[index - 1, 'resistance2'] if index > 0 else 0
                
                diff_ema_5 = df_pd.loc[index, 'close'] - df_pd.loc[index, 'ema_5']
                last_candle_type = df_pd.loc[index - 1, 'candle_type']
                atr = df_pd.loc[index, 'atr']
                rsi = df_pd.loc[index-1, 'rsi_14']
                delta_threshold_1 = 5
                TP_price = 10
                SL_price = 10

                # # Revert True
                if -delta_threshold_1 < last_delta_support1 < delta_threshold_1:
                    if (support_touch_price == current_support and df_pd['close'].iloc[support_touch_time:index].max() - df_pd.loc[index-1, 'close']> 10) or support_touch_price != current_support: 
                        support_touch_price = current_support
                        support_touch_time = index
                        number_touch += 1
                        list_touch_support.append(current_support)
                        list_touch_time_support.append(timestamp)
                        for j in range(index, min(index+50, len(df_pd))):
                            if df_pd.loc[index-1, 'close'] - df_pd.loc[j, 'low'] >= SL_price:
                                break
                            elif df_pd.loc[j, 'high'] - df_pd.loc[index-1, 'close'] >= TP_price:
                                number_revert_true += 1
                                break
                if delta_threshold_1 > last_delta_resistance1 > -delta_threshold_1:
                    if (resistance_touch_price == current_resistance and df_pd.loc[index-1, 'close'] - df_pd['close'].iloc[resistance_touch_time:index].min()> 10) or resistance_touch_price != current_resistance: 
                        resistance_touch_price = current_resistance
                        resistance_touch_time = index
                        number_touch += 1
                        list_touch_resistance.append(current_resistance)
                        list_touch_time_resistance.append(timestamp)
                        for j in range(index, min(index+50, len(df_pd))):
                            if df_pd.loc[index-1, 'close'] - df_pd.loc[j, 'low'] >= SL_price:
                                break
                            elif df_pd.loc[j, 'high'] - df_pd.loc[index-1, 'close'] >= TP_price:
                                number_revert_true += 1
                                break
                
                # === BUY Revert conditions ===
                if (
                    (-delta_threshold_1 < last_delta_support1 < delta_threshold_1 and last_delta_support1 > current_delta_support1 and status_min) and rsi < 40
                ):
                    index_label_support = index
                    df_pd.loc[index, 'labels'] = 1
                        
                    # for j in range(index, min(index+50, len(df_pd))):
                    #     if df_pd.loc[index-1, 'close'] - df_pd.loc[j, 'low'] >= SL_price:
                    #         break
                    #     elif df_pd.loc[j, 'high'] - df_pd.loc[index-1, 'close'] >= TP_price:
                    #         number_logic_true += 1
                    #         break
                # === SELL Revert conditions ===
                if (
                    (delta_threshold_1 > last_delta_resistance1 > -delta_threshold_1 and last_delta_resistance1 < current_delta_resistance1 and status_max) and rsi > 60
                ):
                    index_label_resistance = index
                    df_pd.loc[index, 'labels'] = 0
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
        print(f"Count keylevel: {len(list_support) + len(list_resistance)}")
        print(f"Count revert true: {number_revert_true}")
        print(f"Count touch: {number_touch}")
        print(f"Count logic true: {number_logic_true}")
        # save list support and list resistance to csv
        # Tạo DataFrame cho hỗ trợ
        df_support = pd.DataFrame({
            'time': list_touch_time_support,
            'val': list_touch_support
        })

        # Tạo DataFrame cho kháng cự
        df_resistance = pd.DataFrame({
            'time': list_touch_time_resistance,
            'val': list_touch_resistance
        })

        # Lưu vào file CSV
        df_support.to_csv('support_levels.csv', index=False)
        df_resistance.to_csv('resistance_levels.csv', index=False)
        
if __name__ == "__main__":
        chart_pattern = EAData(
            database="exness_xau_usd_h1",   
            start_time=START_TIME_M15,
            end_time=END_TIME_M15,
        )
        chart_pattern.caculate()