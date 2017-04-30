import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import numpy as np
import pandas as pd
#import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


def rmsle(predicted, actual, size):
    return np.sqrt(np.nansum(np.square(np.log(predicted + 1) - np.log(actual + 1))) / float(size))


columns = [""]
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

print(train_df.columns)
X = train_df[columns]
Y = train_df["price_doc"]

rf = RandomForestRegressor(n_estimators=50,criterion=RMSE)



