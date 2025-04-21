import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load file CSV
def load_and_analyze_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Hiển thị thông tin tổng quan về dữ liệu
    print("\n--- Thông tin dữ liệu ---")
    print(df.info())
    
    # Kiểm tra giá trị null
    print("\n--- Số lượng giá trị thiếu ---")
    print(df.isnull().sum())
    
    # Thống kê mô tả dữ liệu
    print("\n--- Thống kê mô tả ---")
    print(df.describe())
    
    # Trực quan hóa phân phối dữ liệu của từng feature
    df.hist(figsize=(15, 10), bins=50, edgecolor='k')
    plt.suptitle("Phân phối dữ liệu của các đặc trưng")
    plt.show()
    
    # Loại bỏ cột không phải số trước khi vẽ heatmap
    numeric_df = df.select_dtypes(include=['number'])
    
    # Vẽ ma trận tương quan
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("Ma trận tương quan giữa các biến")
    plt.show()
    
    return df

# Đường dẫn file CSV
file_path = "indicator_data_xau_table_m5_2022.csv"
df = load_and_analyze_csv(file_path)