import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from process_data_m15 import *

from xgboost_model import XGBoostModel  # Import the new class
from utils import plot_trading_signals, plot_profit
from tqdm import tqdm

# Đường dẫn thư mục chứa dữ liệu train và file test
TRAIN_DIR = 'data'
TEST_FILE = 'indicator_data_xau_table_h1_2024.csv'
TEST_FILE_M15 = 'indicator_data_xau_table_m15_2024.csv' 
TARGET = 'labels'


def calculate_pl(entry_price, close_price, volume, order_direction):
    if order_direction == "BUY":
        return (close_price - entry_price) * volume * 100
    else:
        return (entry_price - close_price) * volume * 100

def calculate_volume(
    current_balance, stop_loss_delta_price, volume_calculate_by="balance"
):
    if volume_calculate_by == "balance":
        return round(current_balance / 1000 * DEFAULT_VOLUME, 2)
    elif volume_calculate_by == "sl":
        return round(
            (current_balance * BALANCE_PERECENT_LOSS / 100)
            / (stop_loss_delta_price * 100),
            2,
        )

VOLUME_CALCULATE_BY = "balance"
DEFAULT_VOLUME = 0.08
BALANCE_PERECENT_LOSS = 2

def trading_strategy(df_test, y_pred, y_confidence, df_test_m15):
    df_test['predicted_labels'] = y_pred
    df_test['confidence'] = y_confidence

    df_test_m15['profit'] = 0.0
    df_test_m15['trade_outcome'] = None
    df_test_m15['TP'] = 0.0
    df_test_m15['SL'] = 0.0
    df_test_m15['touch_keylevel_time'] = None
    df_test_m15['key_level'] = 0.0

    trades = []
    initial_balance = 1000
    df_test_m15['limit_price'] = 0.0
    list_order = []
    list_id_trade = []
    df_test_m15['predicted_labels'] = 2
    for i in tqdm(range(len(df_test))):
        atr = df_test.loc[i, 'atr']
        k_atr = 2
        entry_price = df_test['close'].iloc[i]
        TP = 10
        SL = 5

        if str(df_test['Date'].iloc[i]) == "2024-01-24 14:30:00":
            print("xxxx")

        if df_test['predicted_labels'].iloc[i] == 1 and df_test['confidence'].iloc[i] > 0.5:
            touch_keylevel_time = df_test['Date'].iloc[i]
            index_entry = df_test_m15.index.get_loc(touch_keylevel_time)
            key_level = df_test['support1'].iloc[i-1] + df_test['close'].iloc[i-1]
            for j in range(index_entry, len(df_test_m15)):
                entry_datetime = df_test_m15.index[j]
                close_price = df_test_m15['close'].iloc[j]
                if -10 < close_price - key_level < 30:
                    current_signal, pattern_type, trading_action = check_partern_1hl_gearing_zone(df_test_m15, entry_datetime, "BUY", touch_keylevel_time , df_test.copy())
                    if pattern_type == 0:
                        current_signal, pattern_type, trading_action = check_partern_2hl_less(df_test_m15, entry_datetime, "BUY", touch_keylevel_time)
                    entry_price_order = df_test_m15['close'].iloc[j]
                else:
                    break
                if pattern_type == 1 or pattern_type == 2:
                    print("BUY - Setting Limit Order")
                    limit_price = current_signal['entry_price'][1]  # Use entry_price from signal
                    
                    # Check if limit price is hit and execute trade
                    for k in range(j, len(df_test_m15)):
                        high_price = df_test_m15['high'].iloc[k]
                        low_price = df_test_m15['low'].iloc[k]
                        close_datetime = df_test_m15.index[k]
                        close_index = df_test_m15.index.get_loc(close_datetime)
                        if close_index - index_entry > 8 * 4 or high_price - entry_price_order > 15:
                            break
                        # Check if limit price is hit for BUY (price must drop to or below limit_price)
                        if low_price <= limit_price:
                            if k not in list_id_trade:
                                df_test_m15['predicted_labels'].iloc[k] = 1
                                df_test_m15['limit_price'].iloc[k] = limit_price
                                df_test_m15['TP'].iloc[k] = limit_price + TP
                                df_test_m15['SL'].iloc[k] = limit_price - SL
                                list_id_trade.append(k)
                                df_test_m15['touch_keylevel_time'].iloc[k] = touch_keylevel_time
                                df_test_m15['key_level'].iloc[k] = key_level
                                
                                # Monitor for TP or SL after limit order is filled
                                for m in range(k + 1, len(df_test_m15)):
                                    if df_test_m15['low'].iloc[m] <= limit_price - SL:
                                        df_test_m15['profit'].iloc[k] = -SL
                                        df_test_m15['trade_outcome'].iloc[k] = 'loss'
                                        trades.append(0)
                                        volume = calculate_volume(initial_balance, SL, VOLUME_CALCULATE_BY)
                                        pl = calculate_pl(limit_price, limit_price - SL, volume, "BUY")
                                        initial_balance += pl
                                        list_order.append({
                                            "id": k,
                                            "order_direction": "BUY",
                                            "entry_date_time": df_test_m15.index[k],
                                            "close_date_time": df_test_m15.index[m],
                                            "current_balance": initial_balance,
                                            "entry_price": limit_price,
                                            "close_price": limit_price - SL,
                                            "volume": volume,
                                            "pl": pl,
                                        })
                                        break
                                    elif df_test_m15['high'].iloc[m] >= limit_price + TP:
                                        df_test_m15['profit'].iloc[k] = TP
                                        df_test_m15['trade_outcome'].iloc[k] = 'win'
                                        trades.append(1)
                                        volume = calculate_volume(initial_balance, TP, VOLUME_CALCULATE_BY)
                                        pl = calculate_pl(limit_price, limit_price + TP, volume, "BUY")
                                        initial_balance += pl
                                        list_order.append({
                                            "id": k,
                                            "order_direction": "BUY",
                                            "entry_date_time": df_test_m15.index[k],
                                            "close_date_time": df_test_m15.index[m],
                                            "current_balance": initial_balance,
                                            "entry_price": limit_price,
                                            "close_price": limit_price + TP,
                                            "volume": volume,
                                            "pl": pl,
                                        })
                                        break
                                break
                    break

        elif df_test['predicted_labels'].iloc[i] == 0 and df_test['confidence'].iloc[i] > 0.5:
            touch_keylevel_time = df_test['Date'].iloc[i]
            key_level = df_test['resistance1'].iloc[i-1] + df_test['close'].iloc[i-1]
            if touch_keylevel_time not in df_test_m15.index:
                index_entry = df_test_m15.index.get_loc(pd.to_datetime(touch_keylevel_time) + pd.Timedelta('60min'))
            else:
                index_entry = df_test_m15.index.get_loc(touch_keylevel_time)
            for j in range(index_entry, len(df_test_m15)):
                entry_datetime = df_test_m15.index[j]
                close_price = df_test_m15['close'].iloc[j]
                if -10 < key_level - close_price < 30:
                    current_signal, pattern_type, trading_action = check_partern_1hl_gearing_zone(df_test_m15, entry_datetime, "SELL", touch_keylevel_time, df_test)
                    entry_price_order = df_test_m15['close'].iloc[j]
                    if pattern_type == 0:
                        current_signal, pattern_type, trading_action = check_partern_2hl_less(df_test_m15, entry_datetime, "SELL", touch_keylevel_time)
                else:
                    break
                if pattern_type == 1 or pattern_type == 2:
                    print("SELL - Setting Limit Order")
                    limit_price = current_signal['entry_price'][1]  # Use entry_price from signal
                    
                    # Check if limit price is hit and execute trade
                    for k in range(j, len(df_test_m15)):
                        high_price = df_test_m15['high'].iloc[k]
                        low_price = df_test_m15['low'].iloc[k]
                        close_datetime = df_test_m15.index[k]
                        close_index = df_test_m15.index.get_loc(close_datetime)
                        if close_index - index_entry > 8 * 4 or entry_price_order - low_price > 15:
                            break
                        
                        # Check if limit price is hit for SELL (price must rise to or above limit_price)
                        if high_price >= limit_price:
                            if k not in list_id_trade:
                                df_test_m15['predicted_labels'].iloc[k] = 0
                                df_test_m15['limit_price'].iloc[k] = limit_price
                                df_test_m15['TP'].iloc[k] = limit_price - TP
                                df_test_m15['SL'].iloc[k] = limit_price + SL
                                df_test_m15['touch_keylevel_time'].iloc[k] = touch_keylevel_time
                                df_test_m15['key_level'].iloc[k] = key_level
                                list_id_trade.append(k)
                                # Monitor for TP or SL after limit order is filled
                                for m in range(k + 1, len(df_test_m15)):
                                    if df_test_m15['high'].iloc[m] >= limit_price + SL:
                                        df_test_m15['profit'].iloc[k] = -SL
                                        df_test_m15['trade_outcome'].iloc[k] = 'loss'
                                        trades.append(0)
                                        volume = calculate_volume(initial_balance, SL, VOLUME_CALCULATE_BY)
                                        pl = calculate_pl(limit_price, limit_price + SL, volume, "SELL")
                                        initial_balance += pl
                                        list_order.append({
                                            "id": k,
                                            "order_direction": "SELL",
                                            "entry_date_time": df_test_m15.index[k],
                                            "close_date_time": df_test_m15.index[m],
                                            "current_balance": initial_balance,
                                            "entry_price": limit_price,
                                            "close_price": limit_price + SL,
                                            "volume": volume,
                                            "pl": pl,
                                        })
                                        break
                                    elif df_test_m15['low'].iloc[m] <= limit_price - TP:
                                        df_test_m15['profit'].iloc[k] = TP
                                        df_test_m15['trade_outcome'].iloc[k] = 'win'
                                        trades.append(1)
                                        volume = calculate_volume(initial_balance, TP, VOLUME_CALCULATE_BY)
                                        pl = calculate_pl(limit_price, limit_price - TP, volume, "SELL")
                                        initial_balance += pl
                                        list_order.append({
                                            "id": k,
                                            "order_direction": "SELL",
                                            "entry_date_time": df_test_m15.index[k],
                                            "close_date_time": df_test_m15.index[m],
                                            "current_balance": initial_balance,
                                            "entry_price": limit_price,
                                            "close_price": limit_price - TP,
                                            "volume": volume,
                                            "pl": pl,
                                        })
                                        break
                                break
                            else:
                                break
                    break
        else:
            df_test['predicted_labels'].iloc[i] = 2

    if trades:
        winrate = sum(trades) / len(trades) * 100
        print(f"Winrate ({len(trades)} trades): {winrate:.2f}%")
    else:
        print("Không có giao dịch nào được thực hiện.")
        winrate = 0.0

    df_test_m15['cumulative_profit'] = df_test_m15['profit'].cumsum()
    return df_test, list_order, df_test_m15

def calculate_performance(backtest_orders, initial_balance, total_bars=0):
    final_balance = initial_balance
    total_net_profit = 0
    profit_factor = 0
    win_rate = 0
    draw_rate = 0
    long_win_rate = 0
    short_win_rate = 0
    total_trades = 0
    long_trades = 0
    short_trades = 0
    gross_profit = 0
    gross_loss = 0
    max_drawdown = {"max_item": None, "list": []}
    largest_profit_trade = {"net": 0, "percentage": 0}
    largest_loss_trade = {"net": 0, "percentage": 0}
    avg_profit_trade = {"net": 0, "percentage": 0}
    avg_loss_trade = {"net": 0, "percentage": 0}
    avg_bars_in_trade = 0
    if len(backtest_orders) == 0:
        return {
            "init_balance": initial_balance,
            "final_balance": final_balance,
            "total_net_profit": total_net_profit,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
        }
    max_dd_value = 0
    max_dd_value_all = 0
    max_dd_list = []
    max_dd_item = None
    loss_chain_count = 0
    max_loss_chain_count = 0
    peak = initial_balance
    for i in range(len(backtest_orders)):
        order = backtest_orders[i]
        if order["order_direction"] == "SELL":
            short_trades += 1
            if order["pl"] > 0:
                gross_profit += order["pl"]
                short_win_rate += 1
                loss_chain_count = 0
            elif order["pl"] < 0:
                gross_loss += order["pl"]
                loss_chain_count += 1
                if loss_chain_count > max_loss_chain_count:
                    max_loss_chain_count = loss_chain_count
            else:
                draw_rate += 1
        elif order["order_direction"] == "BUY":
            long_trades += 1
            if order["pl"] > 0:
                gross_profit += order["pl"]
                long_win_rate += 1
                loss_chain_count = 0
            elif order["pl"] < 0:
                gross_loss += order["pl"]
                loss_chain_count += 1
                if loss_chain_count > max_loss_chain_count:
                    max_loss_chain_count = loss_chain_count
            else:
                draw_rate += 1
        if order["current_balance"] >= peak:
            peak = order["current_balance"]
            if max_dd_item is not None:
                max_dd_list.append(max_dd_item)
            max_dd_item = None
            max_dd_value = 0
            max_loss_chain_count = 0
        else:
            draw_down = (peak - order["current_balance"]) / peak * 100
            if max_dd_item is None:
                max_dd_item = {
                    "value": draw_down,
                    "from": order["entry_date_time"],
                    "to": order["close_date_time"],
                    "number": max_loss_chain_count,
                }
            if draw_down > max_dd_value:
                max_dd_value = draw_down
                max_dd_item["value"] = draw_down
                max_dd_item["to"] = order["close_date_time"]
            max_dd_item["number"] = max_loss_chain_count
            if draw_down > max_dd_value_all:
                max_dd_value_all = draw_down
                max_drawdown["max_item"] = max_dd_item
    if max_dd_item is not None:
        max_dd_list.append(max_dd_item)
    final_balance = backtest_orders[-1]["current_balance"]
    profit_factor = final_balance / initial_balance
    max_drawdown["list"] = max_dd_list
    total_net_profit = final_balance - initial_balance
    return {
        "init_balance": initial_balance,
        "final_balance": final_balance,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


if __name__ == "__main__":
    folder_weight = 'xgboost/weight'
    
    folder_name_model = TEST_FILE.split('/')[-1].split('.')[0].split('_')[-1]
    save_path = os.path.join(folder_weight, folder_name_model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("Đang tải dữ liệu train từ thư mục:", TRAIN_DIR)
    csv_train = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.csv') and 'features' not in f]
    df_train = pd.concat([pd.read_csv(os.path.join(TRAIN_DIR, f)) for f in csv_train])
    df_train = df_train.dropna()

    feature_remove = ['timestamp', 'Date', TARGET, "open", "close", "high", "low", "volume", "ema_5", "ema_25", "ema_34", "ema_89", "ema_50", "ema_200", 'streak_count', 'candle_type',
                     "supply1", "demand1", 'diff_ema_34_89', "diff_ema_34", 'delta_diff_ema_34_89']
    FEATURES = df_train.columns.tolist()
    FEATURES = [f for f in FEATURES if f not in feature_remove]
    np.save(os.path.join(save_path, 'list_features.npy'), FEATURES)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[FEATURES])
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURES, index=df_train.index)
    df_train_scaled = pd.concat([X_train_scaled_df, df_train[[TARGET]]], axis=1)
    
    # Save scaler
    scaler_path = os.path.join(save_path, 'scaler.npy')
    np.save(scaler_path, scaler)

    print("Đang tải dữ liệu test từ file:", TEST_FILE)
    test_df = pd.read_csv(TEST_FILE)
    X_test_scaled = scaler.transform(test_df[FEATURES])
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURES, index=test_df.index)
    y_test = test_df[TARGET]
    
    test_df_m15 = pd.read_csv(TEST_FILE_M15)
    test_df_m15['Date'] = pd.to_datetime(test_df_m15['Date'])
    test_df_m15.set_index('Date', inplace=True)

    print("Đang huấn luyện mô hình XGBoost...")
    
    xgb_model = XGBoostModel(savepath=save_path)
    X_train, y_train, X_val, y_val = xgb_model.split_train_val(df_train_scaled, FEATURES, TARGET)
    model = xgb_model.train(X_train, y_train, X_val, y_val)

    print("Đánh giá mô hình trên tập test:")
    y_pred, y_confidence = xgb_model.evaluate(X_test_scaled_df, y_test)

    print("Áp dụng chiến lược giao dịch...")
    result_df, list_order, df_test_m15 = trading_strategy(test_df.copy(), y_pred, y_confidence, test_df_m15)
    balance = calculate_performance(list_order, initial_balance=1000)

    final_balance = balance["final_balance"]
    profit_factor = balance["profit_factor"]
    max_drawdown = balance["max_drawdown"]["max_item"]["value"] if balance["max_drawdown"]["max_item"] else 0
    print(f"Final Balance: {final_balance}\nProfit Factor: {profit_factor}\nMax Drawdown: {max_drawdown}")

    plot_trading_signals(df_test_m15, result_df, TEST_FILE_M15)
    plot_profit(df_test_m15)
    plt.show()