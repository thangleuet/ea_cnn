from datetime import timedelta
import json
import pandas as pd
from datetime import timedelta
from sqlalchemy import create_engine
from datetime import datetime
import statistics

DB_HOST = "42.96.41.209"
DB_USER = "xttrade"

DB_PASSWORD ="Xttrade1234$"
DB_NAME = "XTTRADE"
db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(db_url, echo=False)


def get_data_m15(current_time) -> str:
    """Get data from database

    Returns:
        string: SQL Query
    """
    start_time = current_time - timedelta(minutes=1500)
    end_time = current_time + timedelta(minutes=50*15)
    sql_query = f"SELECT id, date_time as Date, Open as open, High as high, Low as low, Close as close, Volume as volume, ema34, ema89, output_ta FROM exness_xau_usd_m15 WHERE date_time BETWEEN '{start_time}'  AND '{end_time}' order by date_time"    
    df_m15 = pd.read_sql(sql_query, con=engine)
    df_m15['Date'] = pd.to_datetime(df_m15['Date'])
    df_m15.set_index('Date', inplace=True)
    return df_m15

def calculate_zigzag(df, current_index):
    x_zigzag_etf = json.loads(df.loc[current_index, "output_ta"])["zigzag_data"]["x_zigzag_data"]
    y_zigzag_etf = json.loads(df.loc[current_index, "output_ta"])["zigzag_data"]["y_zigzag_data"]
    type_zigzag_etf = json.loads(df.loc[current_index, "output_ta"])["zigzag_data"]["type_zigzag_data"]
    zigzag = [(timestamp, price, -1 if type_zigzag == 'low' else 1) for timestamp, price, type_zigzag in zip(x_zigzag_etf, y_zigzag_etf, type_zigzag_etf)]
    return zigzag

def last_index_of(lst, value):
    return len(lst) - 1 - lst[::-1].index(value)

def time_to_candle(str_date_start, st_date_end, time_frame="m15"):
    if time_frame == "m5":
        number = 300
    if time_frame == "m15":
        number = 900
    if time_frame == "m30":
        number = 1800
    if time_frame == "h1": 
        number = 3600
    if time_frame == "h4":
        number = 3600*4
    _strart = (
        int(
            datetime.strptime(
                str_date_start, "%Y-%m-%d %H:%M:%S"
            ).timestamp()
        )
        / number
    )
    _end = (
        int(
            datetime.strptime(
                st_date_end, "%Y-%m-%d %H:%M:%S"
            ).timestamp()
        )
        / number
    )
    
    return abs(_end-_strart)

def check_partern_2hl_less(df, current_index, main_entry_direction, touch_keylevel_time):
    x_zigzag_etf = json.loads(df.loc[current_index, "output_ta"])["zigzag_data"]["x_zigzag_data"]
    y_zigzag_etf = json.loads(df.loc[current_index, "output_ta"])["zigzag_data"]["y_zigzag_data"]
    type_zigzag_etf = json.loads(df.loc[current_index, "output_ta"])["zigzag_data"]["type_zigzag_data"]
    list_intersec_etf = json.loads(df.loc[current_index, "output_ta"])["list_intersec"]
    number_intersec = None
    current_price = df.loc[current_index, "close"]
    
    for i in range(len(x_zigzag_etf) -1, -1, -1):
        if pd.to_datetime(x_zigzag_etf[i]) < pd.to_datetime(touch_keylevel_time):
            number_intersec = i-2
            break
    if str(current_index) == "2024-12-05 00:45:00"  or str(current_index) == "2024-02-27 14:00:00":
        print("xxxx")
    # for i in range(len(x_zigzag_etf)):
    #     if int(datetime.strptime(list_intersec_etf[-1]["time"], "%Y-%m-%d %H:%M:%S").timestamp()) < int(datetime.strptime(x_zigzag_etf[i], "%Y-%m-%d %H:%M:%S").timestamp()):
    #         number_intersec = i
    #         break
    
    current_signal = {}
    pattern = None
    pattern_type = 0
    trading_action = "None"
    
    if main_entry_direction == "BUY":
        index_L1 = None
        if number_intersec != None and number_intersec != len(x_zigzag_etf):
            min_element = 100000
            # index_L1 = y_zigzag_htf_htf.index(min(y_zigzag_htf_htf[number_intersec:]))
            for i in range(number_intersec, len(x_zigzag_etf)):
                if index_L1 == None and type_zigzag_etf[i] == "low":
                    if y_zigzag_etf[i] < min_element:
                        min_element = y_zigzag_etf[i]
                    index_L1 = y_zigzag_etf.index(min_element)
                elif index_L1 != None and y_zigzag_etf[i] < min_element:
                    if min_element - y_zigzag_etf[i] > 5:
                        min_element = y_zigzag_etf[i]
                        index_L1 = y_zigzag_etf.index(min_element)
        if (index_L1 != None
                and abs(current_price - y_zigzag_etf[index_L1]) < 10
                and len(x_zigzag_etf) - index_L1 > 2):

                index_max_2 = y_zigzag_etf.index(max(y_zigzag_etf[index_L1:]))
                price_L1 = y_zigzag_etf[index_L1]
                index_L2 = last_index_of(y_zigzag_etf, min(y_zigzag_etf[index_L1 + 1:]))
                price_L2 = y_zigzag_etf[index_L2]
                index_H1 = y_zigzag_etf.index(max(y_zigzag_etf[index_L1:index_L2]))
                price_H1 = y_zigzag_etf[index_H1]
                index_H2 = last_index_of(y_zigzag_etf, max(y_zigzag_etf[index_L2:]))
                price_H2 = y_zigzag_etf[index_H2]
                
                if (abs(min(y_zigzag_etf[index_L1 + 1:]) - price_L1) < (max(y_zigzag_etf[index_L1:index_L2]) - price_L1)/2
                    and index_L2 < len(y_zigzag_etf) - 1
                    and len(x_zigzag_etf) - index_L2 > 1

                    and time_to_candle(x_zigzag_etf[index_L1], x_zigzag_etf[index_H1], time_frame="m15") > 5
                    and abs(price_H1 - price_L1) > 5
                    and abs(price_H2 - price_L2) > 5
                    and abs(price_L1 - max(y_zigzag_etf[index_L1:])) < 20 # giá chạy quá 15 giá thì thôi k vào
                    ):
                    if index_max_2 > index_L2:
                        entry_price = [min(y_zigzag_etf[index_L1 + 1:]) - 1, min(y_zigzag_etf[index_L1 + 1:]) + 2]
                    else:
                        entry_price = [price_L1 - 1, price_L1 + 2]

                    order_type = "LIMIT"
                    tp_price = [entry_price[1], entry_price[1] + 10]
                    pattern = "TWO_HL_LESS"
                    trading_action = "KEYLEVEL_SUPPORT"
                    current_signal = {
                        "signal": "ETF_BUY",
                        "type": "ENTRY",
                        "order_type": order_type,
                        "entry_price": entry_price,
                        "sl_price": entry_price[0],
                        "tp_price": tp_price,
                        "expected_rr": 2.5,
                        "entry_validation_info": {
                            "pa_pattern": pattern,
                            "hl_1": [[], [y_zigzag_etf[-2]], "high", [], [str(x_zigzag_etf[-2])]],
                            "hl_2": [[], [y_zigzag_etf[-2]], "high", [], [str(x_zigzag_etf[-2])]],
                            "signal_date_time": current_index.strftime('%Y-%m-%d %H:%M:%S'),
                            "trading_action_child": trading_action,
                        },
                    }
                    pattern_type = 1
            
    if main_entry_direction == "SELL":
        index_H1 = None
        if number_intersec != None and number_intersec != len(x_zigzag_etf):
            max_element = 0
            for i in range(number_intersec, len(x_zigzag_etf)):
                if index_H1 == None and type_zigzag_etf[i] == "high":
                    if y_zigzag_etf[i] > max_element:
                        max_element = y_zigzag_etf[i]
                        index_H1 = y_zigzag_etf.index(max_element)
                elif index_H1 != None and y_zigzag_etf[i] > max_element:
                    if y_zigzag_etf[i] - max_element > 5:
                        max_element = y_zigzag_etf[i]
                        index_H1 = y_zigzag_etf.index(max_element)
                
        if (index_H1 != None
            and current_price - y_zigzag_etf[index_H1] < 10
            and len(x_zigzag_etf) - index_H1 > 2
            ):
        
            index_min_2 = y_zigzag_etf.index(min(y_zigzag_etf[index_H1:]))
                
            price_H1 = y_zigzag_etf[index_H1]
            index_H2 = last_index_of(y_zigzag_etf, max(y_zigzag_etf[index_H1 + 1:]))
            price_H2 = y_zigzag_etf[index_H2]
            index_L1 = last_index_of(y_zigzag_etf, min(y_zigzag_etf[index_H1:index_H2]))
            price_L1 = y_zigzag_etf[index_L1]
            index_L2 = last_index_of(y_zigzag_etf, min(y_zigzag_etf[index_H2:]))
            price_L2 = y_zigzag_etf[index_L2]

            if time_to_candle(x_zigzag_etf[index_H1], current_index.strftime('%Y-%m-%d %H:%M:%S'), "m15") > 30:
                index_L1_2 = index_min_2
                if (abs(max(y_zigzag_etf[index_H1 + 1:])- price_H1) < (price_H1 - min(y_zigzag_etf[index_H1:index_H2]))/2
                and index_H2 < len(y_zigzag_etf) - 1
                and len(x_zigzag_etf) - index_H2 > 1
    
                and time_to_candle(x_zigzag_etf[index_H1], x_zigzag_etf[index_L1], time_frame="m15") > 5
                and abs(price_H1 - price_L1) > 5
                and abs(price_H2 - price_L2) > 5
                and abs(price_H1 - min(y_zigzag_etf[index_H1:])) < 20
                ):
                        if index_L1_2 > last_index_of(y_zigzag_etf, max(y_zigzag_etf[index_H1 + 1:])):
                            entry_price = [max(y_zigzag_etf[index_H1 + 1:]) + 1, max(y_zigzag_etf[index_H1 + 1:]) - 2]
                        else:
                            entry_price = [price_H1 + 1, price_H1 - 2]

                        order_type = "LIMIT"
                        tp_price = [entry_price[1], entry_price[1] - 10]
                        pattern = "TWO_HL_LESS"
                        trading_action = "KEYLEVEL_RESISTANCE"
                        current_signal = {
                            "signal": "ETF_SELL",
                            "type": "ENTRY",
                            "order_type": order_type,
                            "entry_price": entry_price,
                            "sl_price": entry_price[0],
                            "tp_price": tp_price,
                            "expected_rr": 2.5,
                            "entry_validation_info": {
                                "pa_pattern": pattern,
                                "hl_1": [[], [y_zigzag_etf[-2]], "high", [], [str(x_zigzag_etf[-2])]],
                                "hl_2": [[], [y_zigzag_etf[-2]], "high", [], [str(x_zigzag_etf[-2])]],
                                "signal_date_time": current_index.strftime('%Y-%m-%d %H:%M:%S'),
                                "trading_action_child": trading_action,
                            },
                        }
                        pattern_type = 1
    return current_signal, pattern_type, trading_action

def round_down_to_hour(time_str):
    # Chuyển chuỗi thành đối tượng datetime
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    # Làm tròn xuống giờ (đặt phút và giây = 0)
    dt_rounded = dt.replace(minute=0, second=0, microsecond=0)
    # Chuyển lại về chuỗi nếu cần
    return dt_rounded.strftime("%Y-%m-%d %H:%M:%S")

def get_keylevel_at_time(time_str, data, key_level_type):
    """
    Lấy giá trị support1 tại thời điểm time_str (định dạng giống Date trong self.data).
    Trả về [] nếu không tìm thấy.
    """
    for data_point in reversed(data):
        etf_date = str(["Date"])
        if etf_date == time_str:
            htf_output_ta = json.loads(data_point["htf"]["output_ta"])
            return htf_output_ta["key_level"][key_level_type]
    return []
    
def check_partern_1hl_gearing_zone(df, current_index, main_entry_direction, touch_keylevel_time, df_h1):
    x_zigzag_etf = json.loads(df.loc[current_index, "output_ta"])["zigzag_data"]["x_zigzag_data"]
    y_zigzag_etf = json.loads(df.loc[current_index, "output_ta"])["zigzag_data"]["y_zigzag_data"]
    type_zigzag_etf = json.loads(df.loc[current_index, "output_ta"])["zigzag_data"]["type_zigzag_data"]
    list_intersec_etf = json.loads(df.loc[current_index, "output_ta"])["list_intersec"]
    supports_current = json.loads(df.loc[current_index, "output_ta"])["key_level"]["support1"]
    resistances_current = json.loads(df.loc[current_index, "output_ta"])["key_level"]["resistance1"]
    number_intersec = None   
    # set index
    if not isinstance(df_h1.index, pd.DatetimeIndex):
        df_h1['Date'] = pd.to_datetime(df_h1['Date'])
        df_h1 = df_h1.set_index('Date')
    if str(current_index) == "2024-01-24 14:30:00":
        print("xxxx")
    for i in range(len(x_zigzag_etf) -1, -1, -1):
        if pd.to_datetime(x_zigzag_etf[i]) < pd.to_datetime(touch_keylevel_time):
            number_intersec = i
            break
    
    current_price = df.loc[current_index, "close"]
    
    current_signal = {}
    pattern = None
    pattern_type = 0
    trading_action = "None"
    if main_entry_direction == "BUY":
        index_L1 = None
        if number_intersec != None and number_intersec != len(x_zigzag_etf):
            min_element = 100000
            # index_L1 = y_zigzag_htf_htf.index(min(y_zigzag_htf_htf[number_intersec:]))
            for i in range(number_intersec, len(x_zigzag_etf)):
                if index_L1 == None and type_zigzag_etf[i] == "low":
                    if y_zigzag_etf[i] < min_element:
                        min_element = y_zigzag_etf[i]
                    index_L1 = y_zigzag_etf.index(min_element)
                elif index_L1 != None and y_zigzag_etf[i] < min_element:
                    if min_element - y_zigzag_etf[i] > 5:
                        min_element = y_zigzag_etf[i]
                        index_L1 = y_zigzag_etf.index(min_element)
                        
        if (index_L1 != None
                and abs(current_price - y_zigzag_etf[index_L1]) < 25
                and len(x_zigzag_etf) - index_L1 > 2
            ):
            time_index_previus = round_down_to_hour(x_zigzag_etf[index_L1-1])
            time_current_L2 = round_down_to_hour(str(current_index))
            # list_htf, list_date = get_unique_htf_candles(time_index_previus, time_current_L2)
            list_htf = df_h1.loc[time_index_previus:time_current_L2].values
            list_tight_range = []
            list_boxes = []
            for j, data_point in enumerate(list_htf):
                list_boxes, list_tight_range = caculate_tight_range(j, list_htf, list_tight_range)
                
            if len(list_boxes) > 0:
                box = list_boxes[-1]
                max_box = box[0][1]
                min_box = box[1][1]
                index_box = (max(box[0][0], box[1][0]))
                pass_L2 = False
                
                for n in range(index_box, len(list_htf)):
                    if  pass_L2 == False:
                        for id, y_zz in enumerate(y_zigzag_etf):
                            if y_zigzag_etf[id] < min_box - 3:
                                pass_L2 = True
                                break
                                
                if ( current_price > max_box + 3
                    and current_price > min_box
                    and pass_L2
                ):
                    print("Range box: ", max_box - min_box)
                    entry_price = [min_box - 1, min_box + 2]
                    
                    order_type = "LIMIT"
                    tp_price = [entry_price[1], entry_price[1] + 10]
                    pattern = "ONE_HL_GEARING_ZONE"
                    trading_action = "KEYLEVEL_SUPPORT"
                    current_signal = {
                        "signal": "ETF_BUY",
                        "type": "ENTRY",
                        "order_type": order_type,
                        "entry_price": entry_price,
                        "sl_price": entry_price[0],
                        "tp_price": tp_price,
                        "expected_rr": 2.5,
                        "entry_validation_info": {
                            "pa_pattern": pattern,
                            "hl_1": [[], [y_zigzag_etf[-2]], "high", [], [str(x_zigzag_etf[-2])]],
                            "hl_2": [[], [y_zigzag_etf[-2]], "high", [], [str(x_zigzag_etf[-2])]],
                            "signal_date_time": str(current_index),
                            "trading_action_child": trading_action,
                        },
                    }
                    pattern_type = 2
    
    if main_entry_direction == "SELL":
        index_H1 = None
        if number_intersec != None and number_intersec != len(x_zigzag_etf):
            max_element = 0
            for i in range(number_intersec, len(x_zigzag_etf)):
                if index_H1 == None and type_zigzag_etf[i] == "high":
                    if y_zigzag_etf[i] > max_element:
                        max_element = y_zigzag_etf[i]
                        index_H1 = y_zigzag_etf.index(max_element)
                elif index_H1 != None and y_zigzag_etf[i] > max_element:
                    if y_zigzag_etf[i] - max_element > 5:
                        max_element = y_zigzag_etf[i]
                        index_H1 = y_zigzag_etf.index(max_element)
                            
        if (index_H1 != None
        and abs(current_price - y_zigzag_etf[index_H1]) < 25
        and len(x_zigzag_etf) - index_H1 > 2 
        ):
            time_index_previus = round_down_to_hour(x_zigzag_etf[index_H1-1])
            time_current = str(current_index)
            
            # list_htf, list_date = get_unique_htf_candles(time_index_previus, time_current)
            list_htf = df_h1.loc[time_index_previus:time_current, ["open", "close", "high", "low"]].values
            
            list_tight_range = []
            list_boxes = []
            for j, data_point in enumerate(list_htf):
                list_boxes, list_tight_range = caculate_tight_range(j, list_htf, list_tight_range)
                
            if len(list_boxes) > 0:
                box = list_boxes[-1]
                max_box = box[0][1]
                min_box = box[1][1]
                index_box = (max(box[0][0], box[1][0]))
                pass_H2 = False
                for n in range(index_box, len(list_htf)):
                    if pass_H2 == False:
                        for id, y_zz in enumerate(y_zigzag_etf):
                            if y_zigzag_etf[id] > max_box + 3:
                                pass_H2 = True
                                break
                                
                if (current_price < min_box - 3
                and current_price < max_box
                and pass_H2
                ):
                    entry_price = [max_box + 1, max_box-2]
                    print("Range box: ", max_box - min_box)
                    order_type = "LIMIT"
                    tp_price = [entry_price[1], entry_price[1] + 10]
                    pattern = "ONE_HL_GEARING_ZONE"
                    trading_action = "KEYLEVEL_SUPPORT"
                    current_signal = {
                        "signal": "ETF_SELL",
                        "type": "ENTRY",
                        "order_type": order_type,
                        "entry_price": entry_price,
                        "sl_price": entry_price[0],
                        "tp_price": tp_price,
                        "expected_rr": 2.5,
                        "entry_validation_info": {
                            "pa_pattern": pattern,
                            "hl_1": [[], [y_zigzag_etf[-2]], "high", [], [str(x_zigzag_etf[-2])]],
                            "hl_2": [[], [y_zigzag_etf[-2]], "high", [], [str(x_zigzag_etf[-2])]],
                            "signal_date_time": str(current_index),
                            "trading_action_child": trading_action,
                        },
                    }
                    pattern_type = 2
    return current_signal, pattern_type, trading_action

def get_unique_htf_candles(start_time, end_time, data):
    """Get unique htf candles from self.data within a time range
    
    Args:
        start_time (str): Start time in format "YYYY-MM-DD HH:MM:SS"
        end_time (str): End time in format "YYYY-MM-DD HH:MM:SS"
        
    Returns:
        list: List of unique htf candles and their timestamps
    """
    unique_candles = []
    unique_time = []
    seen_candles = set()  # Use a set for O(1) lookups
    
    # Convert time strings to timestamps once
    start_timestamp = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp()
    end_timestamp = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp()
    
    # Find the start index using binary search (assuming self.data is time-ordered)
    start_idx = 0
    end_idx = len(data) - 1
    start_search_idx = None
    
    # Binary search for start position
    while start_idx <= end_idx:
        mid = (start_idx + end_idx) // 2
        if "htf" not in data[mid] or "Date" not in data[mid]["htf"]:
            # Skip invalid entries by moving forward
            start_idx = mid + 1
            continue
            
        time_mid = data[mid]["htf"]["Date"]
        mid_timestamp = (datetime.strptime(time_mid, "%Y-%m-%d %H:%M:%S").timestamp() 
                        if isinstance(time_mid, str) else time_mid.timestamp())
        
        if mid_timestamp < start_timestamp:
            start_idx = mid + 1
        else:
            end_idx = mid - 1
            start_search_idx = mid
    
    # If we couldn't find a valid starting point, fall back to linear scan
    if start_search_idx is None:
        start_search_idx = 0
    
    # Collect candles from the starting index
    for i in range(start_search_idx, len(data)):
        data_point = data[i]
        if "htf" not in data_point or "Date" not in data_point["htf"]:
            continue
            
        time_current = data_point["htf"]["Date"]
        
        # Convert time_current to timestamp for comparison
        current_timestamp = (
            datetime.strptime(time_current, "%Y-%m-%d %H:%M:%S").timestamp()
            if isinstance(time_current, str)
            else time_current.timestamp()
        )
        
        # Stop if we've passed the end timestamp
        if current_timestamp > end_timestamp:
            break
            
        # Check if the candle is within the time range
        if start_timestamp <= current_timestamp <= end_timestamp:
            # Create candle data tuple (tuples are hashable for set operations)
            candle_data = (
                data_point["htf"]["Open"],
                data_point["htf"]["Close"],
                data_point["htf"]["High"],
                data_point["htf"]["Low"]
            )
            
            # Check if we've already seen this candle data
            if candle_data not in seen_candles:
                seen_candles.add(candle_data)
                unique_candles.append(list(candle_data))
                unique_time.append(data_point["htf"]["Date"])
    
    return unique_candles, unique_time
def caculate_tight_range(i, data, list_tight_range):
    if i > 3:
        point2_high = max(data[i-2][0],data[i-2][1])
        point1_high = max(data[i-1][0],data[i-1][1])
        point0_high = max(data[i][0],data[i][1])

        point2_low = min(data[i-2][0],data[i-2][1])
        point1_low = min(data[i-1][0],data[i-1][1])
        point0_low = min(data[i][0],data[i][1])
        
        list_zigzag_low = [ point2_low, point1_low, point0_low]
        list_zigzag_high = [point2_high, point1_high, point0_high]
        list_points_low = [(i-2, point2_low), (i-1, point1_low), (i, point0_low)]
        list_points_high = [(i-2, point2_high), (i-1, point1_high), (i, point0_high)]
        trc_max  = sum(list_zigzag_high)/len(list_zigzag_high)
        trc_min  = sum(list_zigzag_low)/len(list_zigzag_low)
        number = abs(trc_max - trc_min)
        if len(list_tight_range) == 0:
            if std_deviation_list(list_zigzag_high, number) < 1 and std_deviation_list(list_zigzag_low, number) < 1:
                list_tight_range.append([list_points_high, list_points_low])
        
        elif len(list_tight_range) > 0:
            if list_tight_range[-1][0][-1][1] == point1_high and list_tight_range[-1][1][-1][1] == point1_low:
                
                list_tight_range[-1][0].append((i, point0_high))
                list_tight_range[-1][1].append((i, point0_low))
                
                if std_deviation_list([y[1] for y in list_tight_range[-1][0]], number) < 1 and std_deviation_list([y[1] for y in list_tight_range[-1][1]], number) < 1:
                    pass
                else:
                    list_tight_range[-1][0].pop()
                    list_tight_range[-1][1].pop()
            else:
                if std_deviation_list(list_zigzag_high, number) < 1 and std_deviation_list(list_zigzag_low, number) < 1 and i-11 > list_tight_range[-1][0][-1][0]:
                    list_tight_range.append([list_points_high, list_points_low])
    rt_tight_range = []
    for j in range(len(list_tight_range)):
        x1, x2= list_tight_range[j][0][0][0], list_tight_range[j][0][-1][0]
        y1 = max(list_tight_range[j][0], key=lambda x: x[1])[1]
        y2 = min(list_tight_range[j][1], key=lambda x: x[1])[1]
        
        rt_tight_range.append(((x1, y1), (x2, y2)))
    rt_tight_range_1 = []
    if len(rt_tight_range) > 2:
        rt_tight_range_1 = merge_boxes(rt_tight_range)
    else:
        rt_tight_range_1 = rt_tight_range

    return rt_tight_range_1, list_tight_range

def merge_boxes(boxes):
    merged_boxes = []
    for i in range(len(boxes) - 1):
        box = boxes[i]
        box_1 = boxes[i + 1]
        # iou = self.calculate_iou(box, box_1)
        x1, y1, x2, y2 = box[0][0], box[0][1], box[1][0], box[1][1]
        x3, y3, x4, y4 = box_1[0][0], box_1[0][1], box_1[1][0], box_1[1][1]
        if min(x1, x2) < min(x3, x4) < max(x2 - 2, x1 -2):
            merged_boxes.append(((min(x1, x3, x4, x2), max(y1, y2, y3, y4)), (max(x1, x3, x4, x2), min(y1, y2, y3, y4))))
            i += 1
        else:
            merged_boxes.append(box)
            if i == len(boxes) - 2:
                merged_boxes.append(box_1)
    return merged_boxes

def std_deviation_list(data, number):
    standard_deviation = statistics.stdev(data)
    return standard_deviation