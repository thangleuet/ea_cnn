import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from process_data_m15 import *

from xgboost_model import XGBoostModel  # Import the new class
from tqdm import tqdm

# Đường dẫn thư mục chứa dữ liệu train và file test
TRAIN_DIR = 'data'
TEST_FILE = 'indicator_data_xau_table_h1_2019.csv'
TEST_FILE_H4 = 'indicator_data_xau_table_h4_2019.csv' 
TARGET = 'labels'

VOLUME_CALCULATE_BY = "balance"
DEFAULT_VOLUME = 0.08
BALANCE_PERECENT_LOSS = 2

def trading_strategy(df_test, y_pred, y_confidence, df_test_h4):
    df_test['predicted_labels'] = y_pred
    df_test['confidence'] = y_confidence

    df_test_h4['touch_keylevel_time'] = None
    df_test_h4['key_level'] = 0.0

    list_trade_SL_buy = []
    list_trade_SL_sell = []
    list_trade_TP_buy = []
    list_trade_TP_sell = []
    
    for i in tqdm(range(len(df_test))):
        atr = df_test.loc[i, 'atr']
        k_atr = 2
        entry_price = df_test['close'].iloc[i]
        TP = 10
        SL = 10

        if str(df_test['Date'].iloc[i]) == "2024-01-24 14:30:00":
            print("xxxx")

        if df_test['predicted_labels'].iloc[i] == 1 and df_test['confidence'].iloc[i] > 0.5:
            touch_keylevel_time = df_test['Date'].iloc[i]
            key_level = df_test['support1'].iloc[i-1] + df_test['close'].iloc[i-1]

            if touch_keylevel_time not in df_test_h4.index:
                index_entry = df_test_h4.index.get_indexer([touch_keylevel_time], method='ffill')[0]
            else:
                index_entry = df_test_h4.index.get_loc(touch_keylevel_time) 
            
            delta_ema_h1 = (df_test['ema_34'].iloc[i] - df_test['ema_89'].iloc[i]) > 0
            delta_ema_h4 = (df_test_h4['ema34'].iloc[index_entry] - df_test_h4['ema89'].iloc[index_entry]) > 0
            check_TP = False
            check_SL = False
            type_keylevel = "S"
            # buy
            for j in range(i, len(df_test)):
                if key_level - df_test['low'].iloc[j] > SL:
                    type_trend = "down"
                    if check_TP:
                        break
                    if not check_SL:
                        list_trade_SL_buy.append((touch_keylevel_time, type_keylevel, type_trend, SL, delta_ema_h1, delta_ema_h4))
                        check_SL = True
                    if key_level - df_test['low'].iloc[j] > 1.5*SL:
                        list_trade_SL_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 1.5*SL, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 2*SL:
                        list_trade_SL_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 2*SL, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 2.5*SL:
                        list_trade_SL_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 2.5*SL, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 3*SL:
                        list_trade_SL_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 3*SL, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 3.5*SL:
                        list_trade_SL_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 3.5*SL, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 4*SL:
                        list_trade_SL_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 4*SL, delta_ema_h1, delta_ema_h4)
                        break

                elif df_test['high'].iloc[j] - key_level > TP:
                    type_trend = "up"
                    if check_SL:
                        break
                    if not check_TP:
                        list_trade_TP_buy.append((touch_keylevel_time, type_keylevel, type_trend, TP, delta_ema_h1, delta_ema_h4))
                        check_TP = True
                    if df_test['high'].iloc[j] - key_level > 1.5*TP:
                        list_trade_TP_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 1.5*TP, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 2*TP:
                        list_trade_TP_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 2*TP, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 2.5*TP:
                        list_trade_TP_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 2.5*TP, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 3*TP:
                        list_trade_TP_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 3*TP, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 3.5*TP:
                        list_trade_TP_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 3.5*TP, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 4*TP:
                        list_trade_TP_buy[-1] = (touch_keylevel_time, type_keylevel, type_trend, 4*TP, delta_ema_h1, delta_ema_h4)
                        break

        elif df_test['predicted_labels'].iloc[i] == 0 and df_test['confidence'].iloc[i] > 0.5:
            touch_keylevel_time = df_test['Date'].iloc[i]
            key_level = df_test['resistance1'].iloc[i-1] + df_test['close'].iloc[i-1]
            if touch_keylevel_time not in df_test_h4.index:
                index_entry = df_test_h4.index.get_indexer([touch_keylevel_time], method='ffill')[0]
            else:
                index_entry = df_test_h4.index.get_loc(touch_keylevel_time)
            
            delta_ema_h1 = (df_test['ema_34'].iloc[i] - df_test['ema_89'].iloc[i]) > 0
            delta_ema_h4 = (df_test_h4['ema34'].iloc[index_entry] - df_test_h4['ema89'].iloc[index_entry]) > 0
            check_TP = False
            check_SL = False
            type_keylevel = "R"
            # sell
            for j in range(i, len(df_test)):
                if df_test['high'].iloc[j] - key_level > SL:
                    type_trend = "up"
                    if check_TP:
                        break
                    if not check_SL:
                        list_trade_SL_sell.append((touch_keylevel_time, type_keylevel, type_trend, SL, delta_ema_h1, delta_ema_h4))
                        check_SL = True
                    if df_test['high'].iloc[j] - key_level > 1.5*SL:
                        list_trade_SL_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 1.5*SL, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 2*SL:
                        list_trade_SL_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 2*SL, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 2.5*SL:
                        list_trade_SL_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 2.5*SL, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 3*SL:
                        list_trade_SL_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 3*SL, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 3.5*SL:
                        list_trade_SL_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 3.5*SL, delta_ema_h1, delta_ema_h4)
                    if df_test['high'].iloc[j] - key_level > 4*SL:
                        list_trade_SL_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 4*SL, delta_ema_h1, delta_ema_h4)
                        break
                elif key_level - df_test['low'].iloc[j] > TP:
                    type_trend = "down"
                    if check_SL:
                        break
                    if not check_TP:
                        list_trade_TP_sell.append((touch_keylevel_time, type_keylevel, type_trend, TP, delta_ema_h1, delta_ema_h4))
                        check_TP = True
                    if key_level - df_test['low'].iloc[j] > 1.5*TP:
                        list_trade_TP_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 1.5*TP, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 2*TP:
                        list_trade_TP_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 2*TP, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 2.5*TP:
                        list_trade_TP_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 2.5*TP, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 3*TP:
                        list_trade_TP_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 3*TP, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 3.5*TP:
                        list_trade_TP_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 3.5*TP, delta_ema_h1, delta_ema_h4)
                    if key_level - df_test['low'].iloc[j] > 4*TP:
                        list_trade_TP_sell[-1] = (touch_keylevel_time, type_keylevel, type_trend, 4*TP, delta_ema_h1, delta_ema_h4)
                        break

    # Save trade lists to CSV
    columns = ['time', 'type_keylevel', 'type_trend', 'value', 'delta_ema_h1', 'delta_ema_h4']
    
    df_sl_buy = pd.DataFrame(list_trade_SL_buy, columns=columns)
    df_sl_sell = pd.DataFrame(list_trade_SL_sell, columns=columns)
    df_tp_buy = pd.DataFrame(list_trade_TP_buy, columns=columns)
    df_tp_sell = pd.DataFrame(list_trade_TP_sell, columns=columns)
    
    folder_name_model = TEST_FILE.split('/')[-1].split('.')[0].split('_')[-1]
    df_sl_buy.to_csv(f'sl_buy_trades_{folder_name_model}.csv', index=False)
    df_sl_sell.to_csv(f'sl_sell_trades_{folder_name_model}.csv', index=False)
    df_tp_buy.to_csv(f'tp_buy_trades_{folder_name_model}.csv', index=False)
    df_tp_sell.to_csv(f'tp_sell_trades_{folder_name_model}.csv', index=False)

    # Calculate statistics
    stats = {
        'Type_Keylevel': [],
        'Type_Trend': [],
        'Price_Level': [],
        'Delta_EMA_H1': [],
        'Delta_EMA_H4': [],
        'Count': []
    }
    price_levels = [10, 15, 20, 25, 30, 35, 40]
    for type_keylevel in ['S', 'R']:
        for type_trend in ['up', 'down']:
            for h1 in [True, False]:
                for h4 in [True, False]:
                    for p in price_levels:
                        if type_trend == 'up':
                            if type_keylevel == 'S':
                                count = len([trade for trade in list_trade_TP_buy 
                                           if trade[4] == h1 and trade[5] == h4 and trade[3] == p])
                            else:
                                count = len([trade for trade in list_trade_SL_sell 
                                           if trade[4] == h1 and trade[5] == h4 and trade[3] == p])
                        else:
                            if type_keylevel == 'S':
                                count = len([trade for trade in list_trade_SL_buy 
                                           if trade[4] == h1 and trade[5] == h4 and trade[3] == p])
                            else:
                                count = len([trade for trade in list_trade_TP_sell 
                                           if trade[4] == h1 and trade[5] == h4 and trade[3] == p])
                    
                        stats['Type_Keylevel'].append(f"{type_keylevel}")
                        stats['Type_Trend'].append(f"{type_trend}")
                        stats['Price_Level'].append(p)
                        stats['Delta_EMA_H1'].append(h1)
                        stats['Delta_EMA_H4'].append(h4)
                        stats['Count'].append(count)

    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(f'trade_statistics_{folder_name_model}.csv', index=False)

    # Print summary
    print("\nTrade Statistics Summary:")
    print(df_stats.pivot_table(index=['Delta_EMA_H1', 'Delta_EMA_H4'], 
                             columns='Type_Keylevel', 
                             values='Count', 
                             fill_value=0))

    return df_test, list_trade_SL_buy, list_trade_SL_sell, list_trade_TP_buy, list_trade_TP_sell

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
    
    test_df_h4 = pd.read_csv(TEST_FILE_H4)
    test_df_h4['Date'] = pd.to_datetime(test_df_h4['Date'])
    test_df_h4.set_index('Date', inplace=True)

    print("Đang huấn luyện mô hình XGBoost...")
    
    xgb_model = XGBoostModel(savepath=save_path)
    X_train, y_train, X_val, y_val = xgb_model.split_train_val(df_train_scaled, FEATURES, TARGET)
    model = xgb_model.train(X_train, y_train, X_val, y_val)

    print("Đánh giá mô hình trên tập test:")
    y_pred, y_confidence = xgb_model.evaluate(X_test_scaled_df, y_test)

    print("Áp dụng chiến lược giao dịch...")
    result_df, list_trade_SL_buy, list_trade_SL_sell, list_trade_TP_buy, list_trade_TP_sell = trading_strategy(test_df.copy(), y_pred, y_confidence, test_df_h4)