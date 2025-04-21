import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
import matplotlib as mpl


# Đường dẫn thư mục chứa dữ liệu train và file test
TRAIN_DIR = 'data'
TEST_FILE = 'indicator_data_xau_table_m5_2020.csv'
TARGET = 'labels'

# Danh sách đặc trưng được chọn (phù hợp với cách gán nhãn mới)
# FEATURES = ['stochrsi_d_14', 'body_size', 'delta_diff_ema_34_89', 'diff_ema_89', 'diff_ema_34_89', 'wr_14', 'hour', 'streak_count']
# FEATURES = ['rsi_14', 'macd_diff', 'diff_ema_34', 'diff_ema_89', 'count_ema_34_ema_89', 'RSI_Slope_LR', 'EMA34_Slope_LR', 'cci_20', 'body_size', 'candle_type', 'hour', 'atr']

# 4. Chia dữ liệu train/validation
def split_train_val(df, FEATURES, TARGET, train_ratio=0.8):
    train_size = int(train_ratio * len(df))
    X_train = df[FEATURES][:train_size]
    y_train = df[TARGET][:train_size]
    X_val = df[FEATURES][train_size:]
    y_val = df[TARGET][train_size:]
    return X_train, y_train, X_val, y_val

def compute_class_weights(y_train):
    """Tính trọng số cho các lớp dựa trên tần suất."""
    class_counts = pd.Series(y_train).value_counts()
    total_samples = len(y_train)
    weights = {}
    for cls in class_counts.index:
        weights[cls] = total_samples / class_counts[cls]  # Tỷ lệ nghịch
    # Giới hạn trọng số tối đa
    max_weight = 3
    for cls in weights:
        weights[cls] = min(weights[cls], max_weight)
    # Chuẩn hóa trọng số để hold có trọng số thấp hơn
    weights[2] = 1.0 if 2 in weights else min(weights.values())
    sample_weights = np.array([weights[cls] for cls in y_train])
    return sample_weights

# 5. Huấn luyện mô hình XGBoost
def train_model(X_train, y_train, X_val, y_val):
    sample_weights = compute_class_weights(y_train)
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        max_depth=7,  # Tăng để nắm bắt tương tác
        learning_rate=0.05,  # Giảm để học mượt hơn
        n_estimators=400,  # Tăng số cây
        lambda_=1.5,
        alpha=1,
        eval_metric='mlogloss',
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=True
    )
    return model
# def train_model(X_train, y_train, X_val, y_val):
#     # Áp dụng SMOTE để cân bằng dữ liệu
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#     model = xgb.XGBClassifier(
#         objective='multi:softmax',
#         num_class=3,
#         max_depth=6,
#         learning_rate=0.05,
#         n_estimators=400,
#         lambda_=1.5,
#         alpha=1,
#         eval_metric='mlogloss',
#     )
#     model.fit(
#         X_train_resampled, y_train_resampled,
#         eval_set=[(X_val, y_val)],
#         verbose=True
#     )
#     return model

def plot_confusion_matrix(y_true, y_pred, save_dir='trading_signals'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sell', 'Buy', 'Hold'], yticklabels=['Sell', 'Buy', 'Hold'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    filepath = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Đã lưu confusion matrix: {filepath}")
    plt.show()
    plt.close()
# 6. Đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy trên tập test: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)
    return y_pred

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


def trading_strategy(df_test, y_pred):
    df_test['predicted_labels'] = y_pred

    df_test['profit'] = 0.0
    df_test['trade_outcome'] = None
    df_test['TP'] = 0.0
    df_test['SL'] = 0.0

    trades = []
    initial_balance = 1000

    list_order = []
    for i in range(len(df_test)):
        atr = df_test.loc[i, 'atr']
        k_atr = 1.5  # Hệ số cho ngưỡng giá động
        entry_price = df_test['close'].iloc[i]
        TP = 8
        SL = 5

        if df_test['predicted_labels'].iloc[i] == 1:  # Buy signal
            df_test['TP'].iloc[i] = entry_price + TP
            df_test['SL'].iloc[i] = entry_price - SL

            # Kiểm tra TP/SL từ thời điểm vào lệnh
            for j in range(i+1, len(df_test)):
                if entry_price - df_test['low'].iloc[j] >= SL:
                    df_test['profit'].iloc[i] = -SL
                    df_test['trade_outcome'].iloc[i] = 'loss'
                    trades.append(0)
                    volume = calculate_volume(
                        initial_balance, SL, VOLUME_CALCULATE_BY)
                    pl = calculate_pl(
                        entry_price, entry_price - SL, volume, "BUY")
                    initial_balance += pl
                    list_order.append({
                        "id": i,
                        "order_direction": "BUY",
                        "entry_date_time": df_test['Date'].iloc[i],
                        "close_date_time": df_test['Date'].iloc[j],
                        "current_balance": initial_balance,
                        "entry_price": entry_price,
                        "close_price": entry_price - SL,
                        "volume": volume,
                        "pl": pl,
                    })
                    break
                elif df_test['high'].iloc[j] - entry_price >= TP:
                    df_test['profit'].iloc[i] = TP
                    df_test['trade_outcome'].iloc[i] = 'win'
                    trades.append(1)
                    volume = calculate_volume(
                        initial_balance, TP, VOLUME_CALCULATE_BY)
                    pl = calculate_pl(
                        entry_price, entry_price + TP, volume, "BUY")
                    initial_balance += pl
                    list_order.append({
                        "id": i,
                        "order_direction": "BUY",
                        "entry_date_time": df_test['Date'].iloc[i],
                        "close_date_time": df_test['Date'].iloc[j],
                        "current_balance": initial_balance,
                        "entry_price": entry_price,
                        "close_price": entry_price + TP,
                        "volume": volume,
                        "pl": pl,
                    })
                    break

        elif df_test['predicted_labels'].iloc[i] == 0:  # Sell signal
            df_test['TP'].iloc[i] = entry_price - TP
            df_test['SL'].iloc[i] = entry_price + SL

            # Kiểm tra TP/SL từ thời điểm vào lệnh
            for j in range(i+1, len(df_test)):
                if df_test['high'].iloc[j] - entry_price >= SL:
                    df_test['profit'].iloc[i] = -SL
                    df_test['trade_outcome'].iloc[i] = 'loss'
                    trades.append(0)
                    volume = calculate_volume(
                        initial_balance, SL, VOLUME_CALCULATE_BY)
                    pl = calculate_pl(
                        entry_price, entry_price + SL, volume, "SELL")
                    initial_balance += pl
                    list_order.append({
                        "id": i,
                        "order_direction": "SELL",
                        "entry_date_time": df_test['Date'].iloc[i],
                        "close_date_time": df_test['Date'].iloc[j],
                        "current_balance": initial_balance,
                        "entry_price": entry_price,
                        "close_price": entry_price + SL,
                        "volume": volume,
                        "pl": pl,
                    })

                    break
                elif entry_price - df_test['low'].iloc[j] >= TP:
                    df_test['profit'].iloc[i] = TP
                    df_test['trade_outcome'].iloc[i] = 'win'
                    trades.append(1)
                    volume = calculate_volume(
                        initial_balance, TP, VOLUME_CALCULATE_BY)
                    pl = calculate_pl(
                        entry_price, entry_price - TP, volume, "SELL")
                    initial_balance += pl
                    list_order.append({
                        "id": i,
                        "order_direction": "SELL",
                        "entry_date_time": df_test['Date'].iloc[i],
                        "close_date_time": df_test['Date'].iloc[j],
                        "current_balance": initial_balance,
                        "entry_price": entry_price,
                        "close_price": entry_price - TP,
                        "volume": volume,
                        "pl": pl,
                    })

                    break

    # Tính winrate
    if trades:
        winrate = sum(trades) / len(trades) * 100
        print(f"Winrate ({len(trades)} trades): {winrate:.2f}%")
    else:
        print("Không có giao dịch nào được thực hiện.")
        winrate = 0.0

    # Tính lợi nhuận tích lũy
    df_test['cumulative_profit'] = df_test['profit'].cumsum()

    return df_test, list_order


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
    # largest_profit = sys.maxsize * -1
    # largest_loss = sys.maxsize
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
    
# 8. Vẽ biểu đồ các điểm vào lệnh với nến sử dụng mplfinance và lưu vào thư mục
def plot_trading_signals(df_test,TEST_FILE, save_dir='trading_signals'):
    # Tạo thư mục lưu ảnh nếu chưa tồn tại
    folder_test_name = TEST_FILE.split(".")[0].split("_")[-1]
    folder_test_path = os.path.join(save_dir, folder_test_name)
    if not os.path.exists(folder_test_path):
        os.makedirs(folder_test_path)
    
    # Đảm bảo df_test có các cột cần thiết cho nến
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df_test.columns:
            df_test[col] = pd.read_csv(TEST_FILE)[col]
    
    # Đảm bảo index là DatetimeIndex
    if not isinstance(df_test.index, pd.DatetimeIndex):
        if 'timestamp' in df_test.columns:
            df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
            df_test = df_test.set_index('timestamp')
        else:
            raise ValueError("Index không phải DatetimeIndex và không có cột 'timestamp' để chuyển đổi.")
    
    # Chuẩn bị dữ liệu nến cho mplfinance
    df_test_mpf = df_test.copy()
    
    # Xác định các điểm mua (buy) và bán (sell)
    buy_signals = df_test[df_test['predicted_labels'] == 1].index
    sell_signals = df_test[df_test['predicted_labels'] == 0].index
    
    # Hàm vẽ biểu đồ nến cho một tín hiệu và lưu ảnh
    def plot_candlestick_signal(signal_index, signal_type, start_idx, end_idx, signal_count):
        # Lấy dữ liệu trong khoảng
        plot_data = df_test_mpf.loc[start_idx:end_idx]
        
        # Chuẩn bị dữ liệu cho tín hiệu
        buy_marker = pd.Series(index=plot_data.index, dtype=float)
        sell_marker = pd.Series(index=plot_data.index, dtype=float)
        
        outcome = df_test.loc[signal_index, 'trade_outcome'] if signal_index in df_test.index else None
        profit = df_test.loc[signal_index, 'profit'] if signal_index in df_test.index else 0.0
        
        # Xác định giá vào lệnh
        entry_price = df_test.loc[signal_index, 'close']
        candle_type = df_test.loc[signal_index, 'candle_type'] if 'candle_type' in df_test.columns else 'Unknown'
        
        if signal_index in plot_data.index:
            if signal_type == 'buy':
                buy_marker[signal_index] = df_test.loc[signal_index, 'low'] * 0.999
            else:
                sell_marker[signal_index] = df_test.loc[signal_index, 'high'] * 1.001
        
        # Tạo style cho biểu đồ
        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)
        
        # Tạo danh sách các điểm tín hiệu để thêm vào biểu đồ
        apds = []
        if signal_type == 'buy':
            apds.append(mpf.make_addplot(buy_marker, type='scatter', markersize=100, marker='^', color='green', label=f'Buy Signal {candle_type}'))
        if signal_type == 'sell':
            apds.append(mpf.make_addplot(sell_marker, type='scatter', markersize=100, marker='v', color='red', label=f'Sell Signal {candle_type}'))
        
        # Thêm EMA nếu có
        if 'ema_34' in plot_data.columns and 'ema_89' in plot_data.columns:
            apds.append(mpf.make_addplot(plot_data['ema_34'], type='line', color='blue', label='EMA 34'))
            apds.append(mpf.make_addplot(plot_data['ema_89'], type='line', color='orange', label='EMA 89'))
        
        # Thêm đường ngang cho TP hoặc SL
        tp_price = df_test.loc[signal_index, 'TP']
        sl_price = df_test.loc[signal_index, 'SL']
        
        tp_line = pd.Series(tp_price, index=plot_data.index)
        apds.append(mpf.make_addplot(tp_line, type='line', color='green', linestyle='--', label='Take Profit'))
        sl_line = pd.Series(sl_price, index=plot_data.index)
        apds.append(mpf.make_addplot(sl_line, type='line', color='red', linestyle='--', label='Stop Loss'))
        
        # Vẽ biểu đồ
        fig, ax = mpf.plot(
            plot_data[['open', 'high', 'low', 'close']],
            type='candle',
            style=s,
            title=f'{signal_type.capitalize()} Signal at {signal_index}',
            ylabel='Price',
            addplot=apds,
            figsize=(14, 8),
            returnfig=True
        )
        
        # Thêm chú thích cho TP/SL
        ax = ax[0]
        if outcome:
            outcome_text = f"{outcome.capitalize()} (Profit: {profit:.2f})"
            ax.annotate(
                outcome_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                color='green' if outcome == 'win' else 'red',
                bbox=dict(facecolor='white', alpha=0.8)
            )
        
        # Thêm chú thích giá vào lệnh
        ax.annotate(
            f"Entry: {entry_price:.2f} \n TP: {df_test.loc[signal_index, 'TP']:.2f} \n SL: {df_test.loc[signal_index, 'SL']:.2f}",
            xy=(0.05, 0.80),
            xycoords='axes fraction',
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        folder_image = os.path.join(folder_test_path, "TP") if outcome == 'win' else os.path.join(folder_test_path, "SL")
        if not os.path.exists(folder_image):
            os.makedirs(folder_image)
        
        # Lưu biểu đồ
        signal_time_str = signal_index.strftime('%Y%m%d_%H%M%S')
        filename = f"{signal_type}_signal_{signal_count}_{signal_time_str}.png"
        filepath = os.path.join(folder_image, filename)
        fig.savefig(filepath, bbox_inches='tight')
        print(f"Đã lưu biểu đồ: {filepath}")
        plt.close(fig)
    
    # Đếm số lượng tín hiệu để đánh số file
    buy_count = 1
    sell_count = 1
    
    # Vẽ và lưu từng tín hiệu mua
    for buy_idx in buy_signals:
        idx_pos = df_test.index.get_loc(buy_idx)
        start_pos = max(0, idx_pos - 50)
        end_pos = min(len(df_test), idx_pos + 51)
        start_idx = df_test.index[start_pos]
        end_idx = df_test.index[end_pos - 1]
        plot_candlestick_signal(buy_idx, 'buy', start_idx, end_idx, buy_count)
        buy_count += 1
    
    # Vẽ và lưu từng tín hiệu bán
    for sell_idx in sell_signals:
        idx_pos = df_test.index.get_loc(sell_idx)
        start_pos = max(0, idx_pos - 50)
        end_pos = min(len(df_test), idx_pos + 51)
        start_idx = df_test.index[start_pos]
        end_idx = df_test.index[end_pos - 1]
        plot_candlestick_signal(sell_idx, 'sell', start_idx, end_idx, sell_count)
        sell_count += 1

# 9. Vẽ biểu đồ lợi nhuận
def plot_profit(df_test):
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, df_test['cumulative_profit'], label='Cumulative Profit', color='purple')
    plt.title('Cumulative Profit Over Time')
    plt.xlabel('Time')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_trading_sequence(df_test, TEST_FILE, save_dir='trading_sequences', hours_per_plot=10*24):
    """
    Vẽ biểu đồ các tín hiệu theo chuỗi thời gian liên tục trong khoảng giờ nhất định, với chú thích TP/SL.

    Parameters:
    - df_test: DataFrame chứa dữ liệu test
    - TEST_FILE: tên file test gốc
    - save_dir: thư mục lưu ảnh
    - hours_per_plot: số giờ mỗi ảnh (mặc định 240 giờ = 10 ngày)
    """
    # Đảm bảo cột datetime và index đúng
    if 'timestamp' in df_test.columns:
        df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
        df_test = df_test.set_index('timestamp')
    elif not isinstance(df_test.index, pd.DatetimeIndex):
        raise ValueError("Dữ liệu không có 'timestamp' hoặc index không phải là DatetimeIndex.")

    df_test = df_test.sort_index()

    # Tạo thư mục lưu ảnh
    folder_test_name = TEST_FILE.split(".")[0].split("_")[-1]
    folder_test_path = os.path.join(save_dir, folder_test_name)
    os.makedirs(folder_test_path, exist_ok=True)

    start_time = df_test.index.min()
    end_time = df_test.index.max()

    delta = pd.Timedelta(hours=hours_per_plot)
    current_start = start_time
    seq_count = 1

    while current_start < end_time:
        current_end = current_start + delta
        segment = df_test[(df_test.index >= current_start) & (df_test.index < current_end)]

        if len(segment) < 5:
            current_start = current_end
            continue

        # Marker tín hiệu
        buy_marker = pd.Series(index=segment.index, dtype=float)
        sell_marker = pd.Series(index=segment.index, dtype=float)
        status = pd.Series(index=segment.index, dtype=float)

        buy_marker[segment['predicted_labels'] == 1] = segment['low'][segment['predicted_labels'] == 1] * 0.999
        sell_marker[segment['predicted_labels'] == 0] = segment['high'][segment['predicted_labels'] == 0] * 1.001
        status[segment['trade_outcome'] == "win"] = segment['close'][segment['trade_outcome'] == "win"]
        status[segment['trade_outcome'] == "lose"] = segment['close'][segment['trade_outcome'] == "lose"]

        apds = []
        if buy_marker.any():
            apds.append(mpf.make_addplot(buy_marker, type='scatter', markersize=100, marker='^', color='green', label='Buy'))
        if sell_marker.any():
            apds.append(mpf.make_addplot(sell_marker, type='scatter', markersize=100, marker='v', color='red', label='Sell'))
            
        if status.any():
            apds.append(mpf.make_addplot(status, type='scatter', markersize=100, marker='o', color='blue', label='Status'))

        if 'ema_34' in segment.columns and 'ema_89' in segment.columns:
            apds.append(mpf.make_addplot(segment['ema_34'], color='blue', label='EMA 34'))
            apds.append(mpf.make_addplot(segment['ema_89'], color='orange', label='EMA 89'))

        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)

        fig, ax = mpf.plot(
            segment[['open', 'high', 'low', 'close']],
            type='candle',
            style=s,
            addplot=apds,
            figsize=(14, 8),
            title=f"Trading Signals from {current_start} to {current_end}",
            returnfig=True
        )

        # Lưu biểu đồ
        filename = f"sequence_{seq_count}_{current_start.strftime('%Y%m%d_%H%M')}.png"
        filepath = os.path.join(folder_test_path, filename)
        fig.savefig(filepath, bbox_inches='tight')
        print(f"Đã lưu biểu đồ chuỗi: {filepath}")
        plt.close(fig)

        current_start = current_end
        seq_count += 1


# Main execution
if __name__ == "__main__":
    # Tải và xử lý dữ liệu train
    print("Đang tải dữ liệu train từ thư mục:", TRAIN_DIR)
    csv_train = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.csv') and 'features' not in f]
    df_train = pd.concat([pd.read_csv(os.path.join(TRAIN_DIR, f)) for f in csv_train])
    df_train = df_train.dropna()
    
    feature_remove = ['timestamp', 'Date', TARGET, "open", "close", "high", "low", "volume", "ema_25", "ema_34", "ema_89", "ema_50", "ema_200", 'ema_34_lag_1', 'ema_34_lag_2','cand_type']
    FEATURES = df_train.columns.tolist()
    FEATURES = [f for f in FEATURES if f not in feature_remove]
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[FEATURES])
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURES, index=df_train.index)
    df_train_scaled = pd.concat([X_train_scaled_df, df_train[[TARGET]]], axis=1)
    
    # Chia tập train/validation
    X_train, y_train, X_val, y_val = split_train_val(df_train_scaled, FEATURES, TARGET)
    
    # Tải và xử lý dữ liệu test
    print("Đang tải dữ liệu test từ file:", TEST_FILE)
    test_df = pd.read_csv(TEST_FILE)
    X_test_scaled = scaler.transform(test_df[FEATURES])
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURES, index=test_df.index)
    y_test = test_df[TARGET]
    
    # Huấn luyện mô hình
    print("Đang huấn luyện mô hình XGBoost...")
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Đánh giá mô hình trên tập test
    print("Đánh giá mô hình trên tập test:")
    y_pred = evaluate_model(model, X_test_scaled_df, y_test)
    
    # Xây dựng và đánh giá chiến lược giao dịch
    print("Áp dụng chiến lược giao dịch...")
    result_df, list_order = trading_strategy(test_df.copy(), y_pred)
    balance = calculate_performance(list_order, initial_balance=1000)
    
    final_balance = balance["final_balance"]
    profit_factor = balance["profit_factor"]
    max_drawdown = balance["max_drawdown"]["max_item"]["value"]
    print(f"Final Balance: {final_balance} /n Profit Factor: {profit_factor} /n Max Drawdown: {max_drawdown}")
    
    
    # Vẽ biểu đồ các điểm vào lệnh và thoát lệnh
    # plot_trading_signals(result_df, TEST_FILE)
    plot_trading_sequence(result_df, TEST_FILE)
    plot_profit(result_df)

    # explainer = shap.TreeExplainer(model)

    # # Step 2: Tính shap values
    # shap_values = explainer.shap_values(X_train)  # shap_values là list [class_0, class_1, class_2]
    # shap_values_class_0 = shap_values[:, :, 0]  # class SELL
    # shap_values_class_1 = shap_values[:, :, 1]  # class BUY
    # shap_values_class_2 = shap_values[:, :, 2]  # class NO SIGNAL

    # # Visualize ví dụ: SHAP summary plot for class 1 (BUY)
    # class_names = ['SELL', 'BUY', 'NO SIGNAL']
    # for i in range(3):
    #     print(f"\n🔍 SHAP Summary Plot cho lớp: {class_names[i]}")
    #     shap.summary_plot(shap_values[:, :, i], X_train_scaled_df, plot_type="bar")
    plt.show()
