import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from process_data_m15 import *

from xgboost_model import XGBoostModel  # Import the new class
from tqdm import tqdm

# Đường dẫn thư mục chứa dữ liệu train và file test
TRAIN_DIR = 'data'
TEST_FILE = 'indicator_data_xau_table_h1_2020.csv'
TARGET = 'labels'

VOLUME_CALCULATE_BY = "balance"
DEFAULT_VOLUME = 0.08
BALANCE_PERECENT_LOSS = 2

def trading_strategy(df_test, y_pred, y_confidence):
    df_test['predicted_labels'] = y_pred
    df_test['confidence'] = y_confidence

    list_trade_SL_buy = []
    list_trade_SL_sell = []
    list_trade_TP_buy = []
    list_trade_TP_sell = []
    list_KL_order = []
    
    for i in tqdm(range(len(df_test))):
        atr = df_test.loc[i, 'atr']
        k_atr = 2
        entry_price = df_test['close'].iloc[i]
        TP = 10
        SL = 10

        if str(df_test['Date'].iloc[i]) == "2022-01-24 14:30:00":
            print("xxxx")

        if df_test['predicted_labels'].iloc[i] == 1 and df_test['confidence'].iloc[i] > 0.5:
            keylevel_time = df_test['Date'].iloc[i]
            key_level = df_test['resistance1'].iloc[i] + df_test['close'].iloc[i]
            if key_level in list_KL_order:
                continue
            list_KL_order.append(key_level)
            if df_test['resistance1'].iloc[i] > 5 or df_test['resistance1'].iloc[i] <0:
                print(1)
            delta_ema_h1 = (df_test['ema_34'].iloc[i] - df_test['ema_89'].iloc[i]) > 0
            type_keylevel = "R"
            # buy
            entry_price = df_test['close'].iloc[i]
            entry_time = df_test['Date'].iloc[i]
            for j in range(i+1, len(df_test)):
                if entry_price - df_test['low'].iloc[j] >= SL:
                    list_trade_SL_buy.append((keylevel_time, entry_time, type_keylevel, key_level, SL, delta_ema_h1))
                    break
                elif df_test['high'].iloc[j] - entry_price >= TP:
                    list_trade_TP_buy.append((keylevel_time, entry_time, type_keylevel, key_level, TP, delta_ema_h1))
                    break
 
        elif df_test['predicted_labels'].iloc[i] == 0 and df_test['confidence'].iloc[i] > 0.5:
            keylevel_time = df_test['Date'].iloc[i]
            key_level = df_test['support1'].iloc[i] + df_test['close'].iloc[i]
            if key_level in list_KL_order:
                continue
            list_KL_order.append(key_level)
            if df_test['support1'].iloc[i] < -5 or df_test['support1'].iloc[i] >0:
                print(2)
            delta_ema_h1 = (df_test['ema_34'].iloc[i] - df_test['ema_89'].iloc[i]) > 0
            type_keylevel = "S"
            # sell
            entry_price = df_test['close'].iloc[i]
            entry_time = df_test['Date'].iloc[i]
            for j in range(i+1, len(df_test)):
                if df_test['high'].iloc[j] - entry_price >= SL:
                    list_trade_SL_sell.append((keylevel_time, entry_time, type_keylevel, key_level, SL, delta_ema_h1))
                    break
                elif entry_price - df_test['low'].iloc[j] >= TP:
                    list_trade_TP_sell.append((keylevel_time, entry_time, type_keylevel, key_level, TP, delta_ema_h1))
                    break
    # Save trade lists to CSV
    columns = ['time', 'entry_time', 'type_keylevel', 'key_level', 'value', 'delta_ema_h1']
    
    df_sl_buy = pd.DataFrame(list_trade_SL_buy, columns=columns)
    df_sl_sell = pd.DataFrame(list_trade_SL_sell, columns=columns)
    df_tp_buy = pd.DataFrame(list_trade_TP_buy, columns=columns)
    df_tp_sell = pd.DataFrame(list_trade_TP_sell, columns=columns)
    
    folder_name_model = TEST_FILE.split('/')[-1].split('.')[0].split('_')[-1]
    df_sl_buy.to_csv(f'sl_buy_trades_{folder_name_model}.csv', index=False)
    df_sl_sell.to_csv(f'sl_sell_trades_{folder_name_model}.csv', index=False)
    df_tp_buy.to_csv(f'tp_buy_trades_{folder_name_model}.csv', index=False)
    df_tp_sell.to_csv(f'tp_sell_trades_{folder_name_model}.csv', index=False)
    print(f"Count SL buy: {len(list_trade_SL_buy)}")
    print(f"Count SL sell: {len(list_trade_SL_sell)}")
    print(f"Count TP buy: {len(list_trade_TP_buy)}")
    print(f"Count TP sell: {len(list_trade_TP_sell)}")
    print(f"Total trade: {len(list_trade_SL_buy) + len(list_trade_SL_sell) + len(list_trade_TP_buy) + len(list_trade_TP_sell)}")
    print(f"Winrate: {(len(list_trade_TP_buy) + len(list_trade_TP_sell)) / (len(list_trade_SL_buy) + len(list_trade_SL_sell) + len(list_trade_TP_buy) + len(list_trade_TP_sell))}")

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

    print("Đang huấn luyện mô hình XGBoost...")
    
    xgb_model = XGBoostModel(savepath=save_path)
    X_train, y_train, X_val, y_val = xgb_model.split_train_val(df_train_scaled, FEATURES, TARGET)
    model = xgb_model.train(X_train, y_train, X_val, y_val)

    print("Đánh giá mô hình trên tập test:")
    y_pred, y_confidence = xgb_model.evaluate(X_test_scaled_df, y_test)

    print("Áp dụng chiến lược giao dịch...")
    result_df, list_trade_SL_buy, list_trade_SL_sell, list_trade_TP_buy, list_trade_TP_sell = trading_strategy(test_df.copy(), y_pred, y_confidence)