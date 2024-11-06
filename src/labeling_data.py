import os
import pandas as pd
import numpy as np
import tqdm

def create_labels(self, df, col_name, window_size=11):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels with
                  size = total-(window_size)+1
        """

        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)

        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = int((window_begin + window_end) / 2)

                high_window = df.loc[window_begin : window_end]['high']
                low_window = df.loc[window_begin : window_end]['low']

                max_index = high_window.idxmax()
                min_index = low_window.idxmin()
                max_ = high_window.max()
                min_ = low_window.min()

                min_after = df.loc[max_index : window_end]['low'].min()
                max_after = df.loc[min_index : window_end]['high'].max()

                current_price = df.iloc[window_middle]['close']

                if max_index == window_middle and current_price - min_after > 5:
                    labels[window_middle] = 0
                elif min_index == window_middle and max_after - current_price > 5:
                    labels[window_middle] = 1
                else:
                    labels[window_middle] = 2

            row_counter = row_counter + 1
            pbar.update(1)

        pbar.close()
        return labels

path_data_csv = "data/features_data.csv"
df_raw = pd.read_csv(path_data_csv)
df_raw['labels'] = create_labels(df_raw, 'close')
df_raw.to_csv(path_data_csv, index=False)