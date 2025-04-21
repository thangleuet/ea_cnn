import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV
# Giả định file CSV có cột 'Date' (ngày) và 'C' (giá đóng cửa)
file_path = "indicator_data_eur_table_m5_2024_10.csv"  # Thay đổi đường dẫn tới file của bạn
eurusd = pd.read_csv(file_path)

# Chuyển đổi cột ngày thành định dạng datetime
dates = pd.to_datetime(eurusd['Date'], format="%Y-%m-%d %H:%M:%S")  # Định dạng ngày giống code R
prices = eurusd['close']  # Cột giá đóng cửa

# Tính log returns
returns = np.log(prices / prices.shift(1)).dropna()
returns.index = dates[1:]  # Gán index là ngày, bỏ ngày đầu tiên do shift

# Bước 2: Xác định cửa sổ trượt
window_length = 100  # Độ dài cửa sổ giống code R
forecasts_length = len(returns) - window_length
forecasts = np.zeros(forecasts_length)  # Dự báo lợi nhuận
directions = np.zeros(forecasts_length)  # Hướng: +1 (mua), -1 (bán), 0 (lỗi)

# Bước 3: Dự báo với ARIMA và GARCH
for i in range(forecasts_length):
    # Dữ liệu trong cửa sổ trượt
    roll_returns = returns.iloc[i:i + window_length]
    
    # Tìm tham số ARIMA tối ưu
    final_aic = float('inf')
    final_order = (0, 0, 0)
    for p in range(1, 5):  # p, q từ 1 đến 4 giống code R
        for q in range(1, 5):
            try:
                model = ARIMA(roll_returns, order=(p, 0, q))  # d=0 giống R
                model_fit = model.fit()
                current_aic = model_fit.aic
                if current_aic < final_aic:
                    final_aic = current_aic
                    final_order = (p, 0, q)
                    final_arima = model_fit
            except:
                continue
    
    # Kết hợp ARIMA và GARCH(1,1)
    try:
        # Dùng arch_model với ARMA+GARCH
        spec = arch_model(roll_returns, 
                         mean='AR', lags=final_order[0],  # AR(p)
                         vol='Garch', p=1, q=1,         # GARCH(1,1)
                         dist='skewt')                  # Phân phối skew-t gần giống sged trong R
        fit = spec.fit(disp='off')
        
        # Dự báo 1 ngày tiếp theo
        forecast = fit.forecast(horizon=1)
        next_day_return = forecast.mean.values[-1, 0]
        forecasts[i] = next_day_return
        directions[i] = 1 if next_day_return > 0 else -1  # +1 hoặc -1
    except:
        forecasts[i] = 0  # Gán 0 nếu lỗi, giống R
        directions[i] = 0

# Bước 4: Tạo chuỗi thời gian dự báo
forecasts_ts = pd.Series(forecasts, index=dates[window_length:])
strategy_forecasts = forecasts_ts.shift(1)  # Dịch 1 bước giống Lag trong R
strategy_direction = np.where(strategy_forecasts > 0, 1, np.where(strategy_forecasts < 0, -1, 0))

# Bước 5: Tính lợi nhuận chiến lược
actual_returns = returns[window_length:]
strategy_returns = strategy_direction * actual_returns
strategy_returns.iloc[0] = 0  # Gán giá trị đầu tiên là 0 để loại NA
strategy_curve = strategy_returns.cumsum()  # Lợi nhuận tích lũy

# Bước 6: So sánh với chiến lược mua-và-giữ
longterm_returns = actual_returns
longterm_curve = longterm_returns.cumsum()

# Bước 7: Trực quan hóa
plt.figure(figsize=(10, 6))
plt.plot(strategy_curve, color='green', label='ARIMA+GARCH Strategy')
plt.plot(longterm_curve, color='red', label='Long Term Investing')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

# In kết quả
print("Lợi nhuận tích lũy của chiến lược:", strategy_curve.iloc[-1])
print("Lợi nhuận tích lũy của mua-và-giữ:", longterm_curve.iloc[-1])