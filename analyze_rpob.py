import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
file_path = 'indicator_data_xau_table_m5_2023.csv'  # Thay bằng đường dẫn thực tế
data = pd.read_csv(file_path)

# Định nghĩa các nhãn
label_mapping = {0: 'Sell', 1: 'Buy', 2: 'Hold'}
data['label_name'] = data['labels'].map(label_mapping)

# Danh sách các đặc trưng cần trực quan hóa
features = ['rsi_14', 'macd_diff', 'cci_20', 'diff_ema_34', 'diff_ema_89', 'body_size', 'atr']

# Thiết lập kích thước và layout cho các biểu đồ
plt.figure(figsize=(15, len(features) * 4))

# Vẽ biểu đồ phân bố (histogram) và boxplot cho từng đặc trưng
for i, feature in enumerate(features, 1):
    # Histogram
    plt.subplot(len(features), 2, 2*i-1)
    for label in data['label_name'].unique():
        subset = data[data['label_name'] == label]
        sns.histplot(subset[feature], label=label, kde=True, stat='density', alpha=0.5)
    plt.title(f'Phân bố của {feature} theo nhãn')
    plt.xlabel(feature)
    plt.ylabel('Mật độ')
    plt.legend()

    # Boxplot
    plt.subplot(len(features), 2, 2*i)
    sns.boxplot(x='label_name', y=feature, data=data)
    plt.title(f'Boxplot của {feature} theo nhãn')
    plt.xlabel('Nhãn')
    plt.ylabel(feature)

# Điều chỉnh layout để tránh chồng lấn
plt.tight_layout()
plt.show()