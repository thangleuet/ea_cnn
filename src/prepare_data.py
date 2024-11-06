import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from data_generator import DataGenerator

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

args = sys.argv
print("running stock_cnn.py", args)
pd.options.display.width = 0

print("Tensorflow devices {}".format(tf.test.gpu_device_name()))
folder_path_train = "data"
data_gen_train = DataGenerator(folder_path_train)

folder_path_test = "test"
data_gen_test = DataGenerator(folder_path_test)

# folder_path_val = "val"
# data_gen_test = DataGenerator(folder_path_val)
sys.exit()
