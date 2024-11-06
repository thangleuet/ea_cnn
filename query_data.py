import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from sqlalchemy import create_engine

DB_HOST = "42.96.41.209"
DB_USER = "xttrade"

DB_PASSWORD ="Xttrade1234$"
DB_NAME = "XTTRADE"

start_time_format = "2024-01-01 00:00:00"
end_time_format = "2025-01-01 00:00:00"
name_database = "exness_xau_usd_m15"
sql_query = f"""
    SELECT id, date_time as Date, open as Open, high as High, low as Low, close as Close,
        volume,
        technical_info
    FROM {name_database}
    WHERE date_time BETWEEN '{start_time_format}' AND '{end_time_format}'
    ORDER BY date_time
"""
db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(db_url, echo=False)
print(sql_query)
df_h1 = pd.read_sql(sql_query, con=engine)
df_h1.to_csv("exness_xau_usd_m15_2024_2025.csv", index=False)
print("Query done")