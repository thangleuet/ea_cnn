from datetime import datetime
import json
import os
import time
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
        self.timeframe = "table_m15"

    def get_data(self) -> str:
        """Get data from database

        Returns:
            string: str: SQL Query
        """
        logger.info(f"Database URL: {self.db_url}")
        sql_query = f"SELECT id, indicator_data, output_ta FROM {self.database} WHERE date_time BETWEEN '{self.start_time}'  AND '{self.end_time}' order by date_time"
        logger.info(f"SQL Query: {sql_query}")
        return sql_query
    
    def caculate(self) -> None:
        """Caculate data from database
        - Chart Pattern
        - Save data to Database and backup to csv
        """
        sql_query = self.get_data()
        df_pd = pd.read_sql(sql_query, con=self.engine)
        window_size = 11
        # save to csv
        for index, row in tqdm.tqdm(df_pd.iterrows()):
            id = row["id"]
            indicator_data = eval(row["indicator_data"].replace('NaN', '2'))
            for key, value in indicator_data.items():
                df_pd.loc[index, key] = value
            
            # labeling data
            if index >= window_size - 1:
                window_begin = index - (window_size - 1)
                window_end = index
                window_middle = int((window_begin + window_end) / 2)
                indicator_window = df_pd.loc[window_begin : window_end]["indicator_data"]
                
                high_values = []
                low_values = []

                # Lấy giá trị cao/thấp từ từng chỉ báo trong cửa sổ
                for indicator in indicator_window:
                    indicator = eval(indicator.replace('NaN', '2'))
                    high = indicator["high"]
                    low = indicator["low"]
                    high_values.append(high)
                    low_values.append(low)

                # Xác định giá cao nhất và thấp nhất trong cửa sổ
                max_index = high_values.index(max(high_values)) + window_begin
                min_index = low_values.index(min(low_values)) + window_begin

                # Xác định giá thấp nhất và cao nhất sau max_index và min_index
                min_after = min(low_values[max_index - window_begin :])
                max_after = max(high_values[min_index - window_begin :])

                current_price = df_pd.loc[window_middle, 'close']

                # Gán nhãn tương tự
                if max_index == window_middle and current_price - min_after > 5:
                    df_pd.loc[window_middle, 'labels'] = 0  # SELL
                elif min_index == window_middle and max_after - current_price > 5:
                    df_pd.loc[window_middle, 'labels'] = 1  # BUY
                else:
                    df_pd.loc[window_middle, 'labels'] = 2  # HOLD
                
        
        # remove column indicator_data
        df_pd.drop(columns=["indicator_data"], inplace=True)
        df_pd.drop(columns=["id"], inplace=True)
        df_pd.to_csv(f"indicator_data_{self.timeframe}_{START_TIME_M15.split('-')[0]}.csv", index=False)
            

if __name__ == "__main__":
        chart_pattern = EAData(
            database="exness_xau_usd_m15",
            start_time=START_TIME_M15,
            end_time=END_TIME_M15,
        )
        chart_pattern.caculate()