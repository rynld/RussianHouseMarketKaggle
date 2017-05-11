import os
import operator
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from RussianHouse import RussianHouse
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
macro = pd.read_csv('input/macro.csv')

id_test = test["id"]
rh = RussianHouse()
train,test = rh.transform(train,test)

#corr_20 = rh.corr_plot(train, 20, 'price_doc', 10, 10)

y_train = train["price_doc"]
train.drop("price_doc",inplace=True,axis=1)
x_train = train#.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test#.drop(["id", "timestamp"], axis=1)


xgb_params = {
    "seed":0,
    'eta': 0.03,
    'max_depth': 4,
    'subsample': 0.3,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

total = int(len(train))
val_total = int(len(train)*0.33)

dall = xgb.DMatrix(x_train, y_train)
dtrain = xgb.DMatrix(x_train[:total - val_total], y_train[:total - val_total])
dval = xgb.DMatrix(x_train[- val_total:], y_train[-val_total:])
dtest = xgb.DMatrix(x_test)

partial_model = xgb.train(xgb_params, dtrain,evals=[(dtrain,"train"),(dval,"val")], num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50)

#cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
#plt.show()

num_boost_rounds = partial_model.best_iteration
model = xgb.train(dict(xgb_params), dall, num_boost_round= num_boost_rounds)

# importance = model.get_fscore()
# importance = sorted(importance.items(), key=operator.itemgetter(1))[-20:]
#
# df = pd.DataFrame(importance, columns=['feature', 'fscore'])
# df['fscore'] = df['fscore'] / df['fscore'].sum()
#
# plt.figure()
# df.plot()
# df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# plt.gcf().savefig('feature_importance_xgb.png')

# importance = model.get_fscore()
# importance = sorted(importance.items(), key=operator.itemgetter(1))
# print([ x for x,y in importance if y < 3])

y_predict = model.predict(dtest)
#y_predict = np.exp(y_predict) - 1
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
current_date = datetime.datetime.now()
output.to_csv('outputs/xgbSub{0}-{1}-{2}-{3}.csv'.format(current_date.day,current_date.hour,current_date.minute,current_date.second), index=False)