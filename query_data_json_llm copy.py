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
import requests
import matplotlib.pyplot as plt
import mplfinance as mpf  # For candlestick charts

import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from sqlalchemy import text


DB_HOST = "42.96.41.209"
DB_USER = "xttrade"

DB_PASSWORD ="Xttrade1234$"
DB_NAME = "XTTRADE_REALTIME"

START_TIME_M15 = "2024-01-01 00:00:00"
END_TIME_M15 = "2025-12-31 00:00:00"

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
        self.timeframe = "table_h1"

    def get_data(self) -> str:
        """Get data from database

        Returns:
            string: str: SQL Query
        """
        logger.info(f"Database URL: {self.db_url}")
        sql_query = f"SELECT date_time as Date, Open as open, High as high, Low as low, Close as close, Volume as volume FROM {self.database} WHERE date_time BETWEEN '{self.start_time}'  AND '{self.end_time}' order by date_time"
        logger.info(f"SQL Query: {sql_query}")
        return sql_query  
    
    def caculate(self) -> None:
        """Caculate data from database
        - Chart Pattern
        - Save data to Database and backup to csv
        """
        sql_query = self.get_data()
        df_pd = pd.read_sql(sql_query, con=self.engine)
        
        df_pd['EMA_34'] = df_pd['close'].ewm(span=34, adjust=False).mean()
        df_pd['EMA_89'] = df_pd['close'].ewm(span=89, adjust=False).mean()
        
        folder_json = "json"
        if not os.path.exists(folder_json):
            os.makedirs(folder_json)
        
        # Get the last 100 candles
        # for index, row in df_pd.iterrows():
        index = 8    
        last_100_candles = df_pd[index:index+100]
        if not os.path.exists(f"{folder_json}/{index}"):
            os.makedirs(f"{folder_json}/{index}")
        
        # Gọi phương thức analyze (giả sử nó không ảnh hưởng đến entry point)
        self.analyze(last_100_candles, index, folder_json)
        
        # Lấy giá trị entry point là giá đóng cửa của nến cuối cùng trong last_100_candles
        entry_price = last_100_candles['close'].iloc[-1]
        entry_time = pd.to_datetime(last_100_candles['Date'].iloc[-1])
        
        # Chuẩn bị dữ liệu để vẽ (lấy 115 nến để có thêm ngữ cảnh)
        list_candle_plot = df_pd[index:index+115].copy()
        list_candle_plot['Date'] = pd.to_datetime(list_candle_plot['Date'])
        list_candle_plot.set_index('Date', inplace=True)
        
        # Tính EMA 34 và EMA 89 nếu chưa có
        if 'EMA_34' not in list_candle_plot.columns:
            list_candle_plot['EMA_34'] = list_candle_plot['close'].ewm(span=34, adjust=False).mean()
        if 'EMA_89' not in list_candle_plot.columns:
            list_candle_plot['EMA_89'] = list_candle_plot['close'].ewm(span=89, adjust=False).mean()
        
        # Tạo các đường EMA để vẽ
        ema_34 = mpf.make_addplot(list_candle_plot['EMA_34'], color='orange', width=1.5, label='EMA 34')
        ema_89 = mpf.make_addplot(list_candle_plot['EMA_89'], color='green', width=1.5, label='EMA 89')
        
        # Tìm vị trí của entry_time trong index của list_candle_plot
        # Vì entry_time là thời gian của nến cuối trong last_100_candles, nó sẽ nằm trong list_candle_plot
        entry_series = pd.Series(np.nan, index=list_candle_plot.index)
        entry_index = list_candle_plot.index.get_loc(entry_time)
        entry_series.iloc[entry_index] = entry_price
        
        # Tạo addplot cho entry point (dùng scatter plot)
        entry_plot = mpf.make_addplot(
            entry_series,
            type='scatter',
            markersize=100,  # Kích thước điểm
            marker='^',      # Hình dạng điểm (tam giác hướng lên)
            color='lime',    # Màu xanh lá nhạt cho entry point
            label='Entry Point',
            secondary_y=False
        )
        # Create the candlestick chart
        mpf.plot(
            list_candle_plot,
            type='candle',  # Candlestick chart
            style='charles',  # Chart style (you can use 'binance', 'yahoo', etc.)
            title='Last 100 Candles (XAU/USD)',
            ylabel='Price',
            volume=True,  # Include volume bars
            ylabel_lower='Volume',
            addplot=[ema_34, ema_89, entry_plot],  # Add EMA 34 and EMA 89 to the chart
            savefig=f'json/{index}/last_100_candles_chart.png'  # Save the chart as an image
        )
        logger.info("Saved candlestick chart to last_100_candles_chart.png")
        
    def analyze(self, last_100_candles, index, folder_json):
        
        # Convert DataFrame to dictionary for JSON serialization
        last_100_candles_dict = last_100_candles.to_dict(orient='records')
        
        # Define the output JSON file path
        output_file = f"{folder_json}/{index}/{self.timeframe}_{START_TIME_M15.split('-')[0]}.json"
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(last_100_candles_dict, f, ensure_ascii=False, indent=4, default=str)
            
        # Convert last_100_candles_dict to a JSON string for LLM input
        last_100_candles_json = json.dumps(last_100_candles_dict, default=str)
        
        # Send to LLM
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json",
            },

            data = json.dumps({
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Your role is a technical analysis expert using the Price Action method. You are provided with data for the last 100 candlesticks, each containing: datetime (ISO format), open (float), close (float), high (float), low (float), volume (float). The timeframe of the data is explicitly provided as '1h' (1 hour) unless otherwise specified. Analyze this data using the Price Action method and return the output in JSON format with the following information:
                                - current_price: The closing price of the last candlestick (float).
                                - key_levels: A list of key levels, each as an object with:
                                    - type: 'support', 'resistance', 'supply', 'demand', 'TRC' (trend reversal confirmation), or 'darvas box' (string).
                                        - 'support': Identified if price tests a low level at least 2 times within a 0.5% range without breaking below.
                                        - 'resistance': Identified if price tests a high level at least 2 times within a 0.5% range without breaking above.
                                        - 'supply': Identified by a strong rejection downward (upper wick > 2x body) followed by a rapid price drop, with high volume (> 1.5x average of last 10 candles).
                                        - 'demand': Identified by a strong rejection upward (lower wick > 2x body) followed by a rapid price rise, with high volume (> 1.5x average of last 10 candles).
                                        - 'TRC': Identified by a key level where price breaks and reverses with a confirming candle pattern (e.g., Pin Bar, Engulfing) and volume spike (> 1.5x average).
                                        - 'darvas box': Identified by a range where price oscillates between a high and low for at least 5 candles, with the top/bottom tested at least twice; report the top value (high) here.
                                    - value: The price level (float). For 'darvas box', provide the top value; the bottom can be inferred from context or a separate entry.
                                    - time: The datetime when this level was last tested or confirmed (datetime).
                                - trend: The market trend ('bullish', 'bearish', 'sideways'), determined by higher highs/higher lows (bullish), lower highs/lower lows (bearish), or no clear structure (sideways) over the last 20 candlesticks.
                                - price_structure: An object describing the price structure over the last 30 candlesticks, containing:
                                    - bullish: Boolean, true if a bullish structure is formed (at least one Higher High (HH) or Lower High (LH) compared to previous highs).
                                    - bearish: Boolean, true if a bearish structure is formed (at least one Higher Low (HL) or Lower Low (LL) compared to previous lows).
                                    - EQH: Boolean, true if two consecutive Equal Highs (±0.5% range) are detected.
                                    - EQL: Boolean, true if two consecutive Equal Lows (±0.5% range) are detected.
                                    - pullback: Boolean, true if the price retraces to a key level (support/resistance) or a rejection zone after a strong move (at least 3 candles in the same direction) within the last 10 candlesticks.
                                - candle_pattern: The most notable candlestick pattern at the last candlestick or a key price zone, as an object containing:
                                    - name: The pattern name (string, e.g., 'Pin Bar', 'Bullish Engulfing', 'Doji', or 'none' if none detected). Pin Bar requires a wick > 2x body length; Engulfing requires the current candle body to fully cover the previous candle body.
                                    - direction: The pattern's direction ('bullish', 'bearish', or 'neutral') (string).
                                    - time: The datetime when the pattern occurred (datetime).
                                - momentum: The strength of the trend ('strong', 'moderate', 'weak'). Strong if body > 70% range and 3+ consecutive candles in the same direction; moderate if body > 50% range; weak otherwise.
                                - rejection_zones: A list of strong price rejection zones (via long wicks), each as an object with:
                                    - type: 'upper' (wick > 2x body, rejecting upward move) or 'lower' (wick > 2x body, rejecting downward move) (string).
                                    - value: The price level of the rejection (float).
                                    - time: The datetime of the rejection (datetime).
                                - price_velocity: The average price change rate (% per timeframe unit, float), calculated over the last 10 candlesticks and normalized to the specified timeframe.
                                - chart_pattern: The detected price pattern (string, e.g., 'head and shoulders', 'double top', 'ascending triangle', or 'none' if no clear pattern), based on key_levels and price structure over the last 50 candlesticks
                            """
                            },
                            {
                                "type": "text",
                                "text": last_100_candles_json
                            }
                        ]
                    }
                ]
            })
        )
        # Print the LLM response
        llm_response = response.json()
        print(llm_response)
        
        # Save LLM response to a JSON file
        response_file = f"{folder_json}/{index}/llm_response.json"
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(llm_response, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved LLM response to {response_file}")
            
if __name__ == "__main__":
        chart_pattern = EAData(
            database="exness_xau_usd_h1",
            start_time=START_TIME_M15,
            end_time=END_TIME_M15,
        )
        chart_pattern.caculate()