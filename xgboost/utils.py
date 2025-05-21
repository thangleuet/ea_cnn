import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from process_data_m15 import calculate_zigzag, get_data_m15
from datetime import timedelta

def plot_candlestick_signal(
    df_plot, signal_index, signal_type, start_idx, end_idx, signal_count,
    save_dir, outcome=None, profit=0.0, entry_price=None, tp_price=None, sl_price=None,
    candle_type='Unknown', timeframe='M5', zigzag=None, key_level_price=None
):
    """
    Plot candlestick chart for a trading signal and save the figure.

    Args:
        df_plot (pd.DataFrame): DataFrame containing OHLC data and optional indicators.
        signal_index: Index of the signal (datetime or index).
        signal_type (str): 'buy' or 'sell'.
        start_idx: Start index for plotting window.
        end_idx: End index for plotting window.
        signal_count (int): Counter for naming the output file.
        save_dir (str): Directory to save the plot.
        outcome (str, optional): Trade outcome ('win', 'loss', 'no_trade').
        profit (float, optional): Profit from the trade.
        entry_price (float, optional): Entry price of the trade.
        tp_price (float, optional): Take-profit price.
        sl_price (float, optional): Stop-loss price.
        candle_type (str, optional): Type of candle (e.g., 'Bullish Engulfing').
        timeframe (str, optional): Timeframe of the data (e.g., 'M5', 'M15').
        zigzag (pd.Series, optional): ZigZag values to plot.

    Returns:
        None
    """
    # Ensure save directory exists
    folder_image = os.path.join(save_dir, "TP" if outcome == 'win' else "SL")
    os.makedirs(folder_image, exist_ok=True)
    # Slice data for the plotting window
    plot_data = df_plot.loc[start_idx:end_idx]

    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in plot_data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Prepare signal markers
    buy_marker = pd.Series(index=plot_data.index, dtype=float)
    sell_marker = pd.Series(index=plot_data.index, dtype=float)

    if signal_index in plot_data.index:
        if signal_type == 'buy':
            buy_marker[signal_index] = plot_data.loc[signal_index, 'low'] * 0.999
        elif signal_type == 'sell':
            sell_marker[signal_index] = plot_data.loc[signal_index, 'high'] * 1.001

    # Create market style
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)

    # Prepare additional plots (signals, EMAs, TP/SL lines, ZigZag)
    apds = []
    if signal_type == 'buy' and buy_marker.notna().any():
        apds.append(mpf.make_addplot(buy_marker, type='scatter', markersize=100, marker='^', color='green', label=f'Buy Signal {candle_type}'))
    if signal_type == 'sell' and sell_marker.notna().any():
        apds.append(mpf.make_addplot(sell_marker, type='scatter', markersize=100, marker='v', color='red', label=f'Sell Signal {candle_type}'))

    # Add EMAs if available
    if 'ema34' in plot_data.columns and 'ema89' in plot_data.columns:
        apds.append(mpf.make_addplot(plot_data['ema34'], type='line', color='blue', label='EMA 34'))
        apds.append(mpf.make_addplot(plot_data['ema89'], type='line', color='orange', label='EMA 89'))

    # Add TP/SL lines if provided
    if tp_price is not None:
        tp_line = pd.Series(tp_price, index=plot_data.index)
        apds.append(mpf.make_addplot(tp_line, type='line', color='green', linestyle='--', label='Take Profit'))
    if sl_price is not None:
        sl_line = pd.Series(sl_price, index=plot_data.index)
        apds.append(mpf.make_addplot(sl_line, type='line', color='red', linestyle='--', label='Stop Loss'))
    if key_level_price is not None:
        key_level_line = pd.Series(key_level_price, index=plot_data.index)
        apds.append(mpf.make_addplot(key_level_line, type='line', color='yellow', linestyle='--', label='Key Level'))

    # Add ZigZag if provided
    if zigzag is not None:
        if isinstance(zigzag, pd.Series):
            zigzag_plot = zigzag.loc[start_idx:end_idx]
            if zigzag_plot.notna().any():
                apds.append(mpf.make_addplot(zigzag_plot, type='line', color='purple', label='ZigZag', width=1.5))
        elif isinstance(zigzag, list):
            # Convert list of (index, price, direction) to a Series for plotting
            zigzag_series = pd.Series(index=plot_data.index, dtype=float)
            valid_points = [(idx, price) for idx, price, _ in zigzag if idx in plot_data.index]
            if len(valid_points) >= 2:
                # Interpolate between consecutive ZigZag points
                for i in range(len(valid_points)-1):
                    idx1, price1 = valid_points[i]
                    idx2, price2 = valid_points[i+1]
                    # Linear interpolation between idx1 and idx2
                    steps = plot_data.index.get_loc(idx2) - plot_data.index.get_loc(idx1)
                    if steps > 0:
                        for j in range(steps + 1):
                            interp_idx = plot_data.index[plot_data.index.get_loc(idx1) + j]
                            interp_price = price1 + (price2 - price1) * j / steps
                            zigzag_series[interp_idx] = interp_price
                apds.append(mpf.make_addplot(zigzag_series, type='line', color='purple', label='ZigZag', width=1.5))
                
    # Plot the candlestick chart
    fig, ax = mpf.plot(
        plot_data[['open', 'high', 'low', 'close']],
        type='candle',
        style=s,
        title=f'{signal_type.capitalize()} Signal at {signal_index} ({timeframe})',
        ylabel='Price',
        addplot=apds,
        figsize=(14, 8),
        returnfig=True
    )

    # Add annotations
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

    if entry_price is not None and tp_price is not None and sl_price is not None:
        ax.annotate(
            f"Entry: {entry_price:.2f}\nTP: {tp_price:.2f}\nSL: {sl_price:.2f}",
            xy=(0.05, 0.80),
            xycoords='axes fraction',
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    # Save the plot
    signal_time_str = signal_index.strftime('%Y%m%d_%H%M%S')
    filename = f"{signal_type}_signal_{signal_count}_{signal_time_str}_{timeframe}.png"
    filepath = os.path.join(folder_image, filename)
    fig.savefig(filepath, bbox_inches='tight')
    print(f"Đã lưu biểu đồ ({timeframe}): {filepath}")
    plt.close(fig)


def plot_trading_signals(df_test_m15, df_test_h1, TEST_FILE, save_dir='trading_signals'):
    # Create directory for plots
    folder_test_name = TEST_FILE.split(".")[0].split("_")[-1]
    folder_test_path = os.path.join(save_dir, folder_test_name)
    os.makedirs(folder_test_path, exist_ok=True)
    if not isinstance(df_test_h1.index, pd.DatetimeIndex):
        df_test_h1['Date'] = pd.to_datetime(df_test_h1['Date'])
        df_test_h1 = df_test_h1.set_index('Date')

    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df_test_m15.columns:
            df_test_m15[col] = pd.read_csv(TEST_FILE)[col]

    # Ensure DatetimeIndex
    if not isinstance(df_test_m15.index, pd.DatetimeIndex):
        if 'timestamp' in df_test_m15.columns:
            df_test_m15['timestamp'] = pd.to_datetime(df_test_m15['timestamp'])
            df_test_m15 = df_test_m15.set_index('timestamp')
        else:
            raise ValueError("Index không phải DatetimeIndex và không có cột 'timestamp' để chuyển đổi.")

    # Identify buy and sell signals
    buy_signals = df_test_m15[df_test_m15['predicted_labels'] == 1].index
    sell_signals = df_test_m15[df_test_m15['predicted_labels'] == 0].index

    # Plot signals for df_test and df_m15
    buy_count = 1
    sell_count = 1

    for buy_idx in buy_signals:
        idx_pos = df_test_m15.index.get_loc(buy_idx)
        start_pos = max(10, idx_pos - 100)
        end_pos = min(len(df_test_m15)-1, idx_pos + 101)
        start_idx = df_test_m15.index[start_pos]
        end_idx = df_test_m15.index[end_pos - 1]

        # Plot for df_test (M5)
        outcome = df_test_m15.loc[buy_idx, 'trade_outcome'] if buy_idx in df_test_m15.index else None
        profit = df_test_m15.loc[buy_idx, 'profit'] if buy_idx in df_test_m15.index else 0.0
        entry_price = df_test_m15.loc[buy_idx, 'close']
        tp_price = df_test_m15.loc[buy_idx, 'TP']
        sl_price = df_test_m15.loc[buy_idx, 'SL']
        candle_type = df_test_m15.loc[buy_idx, 'candle_type'] if 'candle_type' in df_test_m15.columns else 'Unknown'
        touch_keylevel_time = df_test_m15.loc[buy_idx, 'touch_keylevel_time']
        key_level_price = df_test_m15.loc[buy_idx, 'key_level']
        
        zigzag = calculate_zigzag(df_test_m15, buy_idx)

        plot_candlestick_signal(
            df_plot=df_test_m15,
            signal_index=buy_idx,
            signal_type='buy',
            start_idx=start_idx,
            end_idx=end_idx,
            signal_count=buy_count,
            save_dir=folder_test_path,
            outcome=outcome,
            profit=profit,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            candle_type=candle_type,
            timeframe='M15',
            zigzag=zigzag,
            key_level_price=key_level_price
        )

        # Plot for df_m15 with ZigZag
        touch_keylevel_index = df_test_h1.index.get_loc(touch_keylevel_time)
        h1_start_idx = max(touch_keylevel_index - 100, 0)
        h1_end_idx = min(touch_keylevel_index + 100, len(df_test_h1)-1)
        h1_start_time = df_test_h1.index[h1_start_idx]
        h1_end_time = df_test_h1.index[h1_end_idx]
        
        plot_candlestick_signal(
            df_plot=df_test_h1,
            signal_index=pd.to_datetime(touch_keylevel_time),
            signal_type='buy',
            start_idx=h1_start_time,
            end_idx=h1_end_time,
            signal_count=buy_count,
            save_dir=os.path.join(folder_test_path, 'H1'),
            outcome=outcome,
            profit=profit,
            entry_price=df_test_h1["close"].loc[touch_keylevel_time],
            tp_price=tp_price,
            sl_price=sl_price,
            candle_type=candle_type,
            timeframe='H1',
            key_level_price=key_level_price
        )
        buy_count += 1

    for sell_idx in sell_signals:
        idx_pos = df_test_m15.index.get_loc(sell_idx)
        start_pos = max(10, idx_pos - 100)
        end_pos = min(len(df_test_m15)-1, idx_pos + 100)
        start_idx = df_test_m15.index[start_pos]
        end_idx = df_test_m15.index[end_pos - 1]

        # Plot for df_test (M5)
        outcome = df_test_m15.loc[sell_idx, 'trade_outcome'] if sell_idx in df_test_m15.index else None
        profit = df_test_m15.loc[sell_idx, 'profit'] if sell_idx in df_test_m15.index else 0.0
        entry_price = df_test_m15.loc[sell_idx, 'close']
        tp_price = df_test_m15.loc[sell_idx, 'TP']
        sl_price = df_test_m15.loc[sell_idx, 'SL']
        candle_type = df_test_m15.loc[sell_idx, 'candle_type'] if 'candle_type' in df_test_m15.columns else 'Unknown'
        touch_keylevel_time = df_test_m15.loc[sell_idx, 'touch_keylevel_time']
        key_level_price = df_test_m15.loc[sell_idx, 'key_level']

        zigzag = calculate_zigzag(df_test_m15, sell_idx)
        plot_candlestick_signal(
            df_plot=df_test_m15,
            signal_index=sell_idx,
            signal_type='sell',
            start_idx=start_idx,
            end_idx=end_idx,
            signal_count=sell_count,
            save_dir=folder_test_path,
            outcome=outcome,
            profit=profit,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            candle_type=candle_type,
            timeframe='M15',
            zigzag=zigzag,
            key_level_price=key_level_price
        )

        # Plot for df_m15 with ZigZag
        touch_keylevel_index = df_test_h1.index.get_loc(touch_keylevel_time)
        h1_start_idx = max(touch_keylevel_index - 100, 0)
        h1_end_idx = min(touch_keylevel_index + 100, len(df_test_h1)-1)
        h1_start_time = df_test_h1.index[h1_start_idx]
        h1_end_time = df_test_h1.index[h1_end_idx]
        
        plot_candlestick_signal(
            df_plot=df_test_h1,
            signal_index=pd.to_datetime(touch_keylevel_time),
            signal_type='sell',
            start_idx=h1_start_time,
            end_idx=h1_end_time,
            signal_count=sell_count,
            save_dir=os.path.join(folder_test_path, 'H1'),
            outcome=outcome,
            profit=profit,
            entry_price=df_test_h1["close"].loc[touch_keylevel_time],
            tp_price=tp_price,
            sl_price=sl_price,
            candle_type=candle_type,
            timeframe='H1',
            key_level_price=key_level_price
        )
        sell_count += 1

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
    if 'timestamp' in df_test.columns:
        df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
        df_test = df_test.set_index('timestamp')
    elif not isinstance(df_test.index, pd.DatetimeIndex):
        raise ValueError("Dữ liệu không có 'timestamp' hoặc index không phải là DatetimeIndex.")

    df_test = df_test.sort_index()
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

        buy_marker = pd.Series(index=segment.index, dtype=float)
        sell_marker = pd.Series(index=segment.index, dtype=float)
        status = pd.Series(index=segment.index, dtype=float)

        buy_marker[segment['predicted_labels'] == 1] = segment['limit_price'][segment['predicted_labels'] == 1]
        sell_marker[segment['predicted_labels'] == 0] = segment['limit_price'][segment['predicted_labels'] == 0]
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
            apds.append(mpf.make_addplot(segment['ema_5'], color='yellow', label='EMA 5'))

        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)

        fig, ax = mpf.plot(
            segment[['open', 'high', 'low', 'close']],
            type='candle',
            style=s,
            addplot=apds,
            figsize=(14, 8),
            title=f"Trading Signals and Key Levels from {current_start} to {current_end}",
            returnfig=True
        )

        filename = f"sequence_{seq_count}_{current_start.strftime('%Y%m%d_%H%M')}.png"
        filepath = os.path.join(folder_test_path, filename)
        fig.savefig(filepath, bbox_inches='tight')
        print(f"Đã lưu biểu đồ chuỗi: {filepath}")
        plt.close(fig)

        current_start = current_end
        seq_count += 1