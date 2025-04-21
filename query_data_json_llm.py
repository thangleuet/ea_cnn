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
        self.timeframe = "table_m5"

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
        # Get the last 100 candles
        # for index, row in df_pd.iterrows():
        index = 31
        last_100_candles = df_pd[index:index+100]
        self.analyze(last_100_candles, index)
        
    def analyze(self, last_100_candles, index):
        
        # Convert DataFrame to dictionary for JSON serialization
        last_100_candles_dict = last_100_candles.to_dict(orient='records')
        folder_json = "json"
        if not os.path.exists(folder_json):
            os.makedirs(folder_json)
        
        # Define the output JSON file path
        output_file = f"{folder_json}/{index}_{self.timeframe}_{START_TIME_M15.split('-')[0]}.json"
        
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
                                - entry_price: suggested optimal entry price  (each as an object), determined using Price Action and EMA 34/89 with conditions Follow Trend: Propose when price reacts (pullback or rejection) at EMA 89 in the direction of the trend (bullish: price above 89 with bullish candle pattern; bearish: price below 89 with bearish candle pattern), confirmed by volume (> 1.5x average of last 10 candles) or Counter Trend: Propose when price forms a reversal structure (e.g., TRC, double top/bottom, or special structure like 2HL/EQH/EQL) near EMA 34 or EMA 89, with a confirming candle pattern (e.g., Pin Bar, Engulfing) and volume spike (> 1.5x average):
                                    - type: 'follow_trend' or 'counter_trend' (string).
                                    - value: The suggested entry price (float).
                                    - time: The datetime of the entry signal (datetime).

                                    - TP: The take-profit price for each entry (float):
                                        - Set at the nearest key level (support/resistance/supply/demand) with strength >= 2, prioritizing resistance (if buying) or support (if selling).
                                        - Set to null if no clear signal or no suitable level.
                                    - SL: The stop-loss price for each entry (float):
                                        - Set below the nearest support/demand or EMA (if buying) or above the nearest resistance/supply or EMA (if selling), prioritizing rejection_zones or significant wicks.
                                        - Ensure a minimum R:R of 1:1.5; if not possible, adjust SL to the nearest wick but maintain R:R > 1:1.
                                        - Set to null if no clear signal.
                                    - Set to empty list [] if no clear signal is present."""
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
        response_file = f"{folder_json}/{index}_llm_response.json"
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(llm_response, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved LLM response to {response_file}")
        
        # Plot the last 100 candles using mplfinance
        # Ensure the 'Date' column is in datetime format and set as index
        last_100_candles['Date'] = pd.to_datetime(last_100_candles['Date'])
        last_100_candles.set_index('Date', inplace=True)
        
        # Create the candlestick chart
        mpf.plot(
            last_100_candles,
            type='candle',  # Candlestick chart
            style='charles',  # Chart style (you can use 'binance', 'yahoo', etc.)
            title='Last 100 Candles (XAU/USD M5)',
            ylabel='Price',
            volume=True,  # Include volume bars
            ylabel_lower='Volume',
            savefig='last_100_candles_chart.png'  # Save the chart as an image
        )
        logger.info("Saved candlestick chart to last_100_candles_chart.png")
            
if __name__ == "__main__":
        chart_pattern = EAData(
            database="exness_xau_usd_m5",
            start_time=START_TIME_M15,
            end_time=END_TIME_M15,
        )
        chart_pattern.caculate()