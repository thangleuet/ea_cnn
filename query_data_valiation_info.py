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

START_TIME = "2024-12-24 00:00:00"
END_TIME = "2024-12-25 00:00:00"

action = {"MONITORING_TREND_UP_START_TREND": 0, "MONITORING_TREND_UP_IN_TREND": 1, "MONITORING_TREND_DOWN_START_TREND": 2, "MONITORING_TREND_DOWN_IN_TREND": 3}
signal = {"ETF_BUY": 1, "ETF_SELL": 0, "ETF_BUY ": 1, "ETF_SELL ": 0}

class ValiationInfo:
    def __init__(self, database: str, database_ea_id: str, start_time: str, end_time: str) -> None:
        self.database = database
        self.database_ea_id = database_ea_id
        self.start_time = start_time
        self.end_time = end_time
        # Create a MySQL database connection with pymysql
        self.db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
        self.engine = create_engine(self.db_url, echo=False)
        
    def get_data(self, id_eas) -> str:
        """Get data from database

        Returns:
            string: str: SQL Query
        """
        logger.info(f"Database URL: {self.db_url}")
        id_eas_str = ",".join(map(str, id_eas))
        sql_query = f"""
            SELECT open_order_id, order_type, order_direction, entry_price, entry_date_time, close_price, close_date_time, volume, tp_price, sl_price, pl, entry_validation_info 
            FROM {self.database} 
            WHERE ea_id IN ({id_eas_str}) 
            AND created_at BETWEEN '{self.start_time}' AND '{self.end_time}' 
            ORDER BY created_at
        """
        logger.info(f"SQL Query: {sql_query}")
        return sql_query
    
    def get_id_ea(self):
        sql_query = f"""
            SELECT id, ea_name 
            FROM {self.database_ea_id} 
            WHERE ea_name = 'SMC_Ver3' 
            AND created_at BETWEEN '{self.start_time}' AND '{self.end_time}' 
            ORDER BY created_at
        """
        df_id_name = pd.read_sql(sql_query, con=self.engine)
        id_eas = df_id_name['id'].tolist()
        return id_eas
    
    def caculate(self) -> None:
        id_eas = self.get_id_ea()
        sql_query = self.get_data(id_eas)
        df_pd = pd.read_sql(sql_query, con=self.engine)
        df_pd = df_pd.drop_duplicates()
        df_feature = pd.DataFrame()
        for index, row in df_pd.iterrows():
            entry_validation_info = eval(row["entry_validation_info"].replace('null', '2'). replace('false', '0').replace('true', '1'))
            entry_time = row["entry_date_time"]
            input_AI = entry_validation_info["input_AI"]
            for key, value in input_AI.items():
                if key == "diff_x_time_1" or key == "diff_x_time_2":
                    value = (pd.to_datetime(entry_time) - pd.to_datetime(value)).total_seconds()//3600
                if key == "action":
                    if value in action:
                        value = action[value]
                if key == "signal":
                    if value in signal:
                        value = signal[value]
                df_feature.loc[index, key] = value
            
            #label
            pl = row["pl"]
            df_feature.loc[index, "labels"] = 0 if pl < 0 else 1
        print(df_feature["labels"].value_counts())    
        df_feature.to_csv(f"indicator_data_xau_valiation.csv", index=False)
        
if __name__ == "__main__":
        chart_pattern = ValiationInfo(
            database="backtest_trading_signal",
            database_ea_id = "ea_backtest_result",
            start_time=START_TIME,
            end_time=END_TIME,
        )
        chart_pattern.caculate()