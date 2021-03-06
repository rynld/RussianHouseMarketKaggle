import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Pred_na import predict_missing_variable
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
macro = pd.read_csv('input/macro.csv')
id_test = test.id

# train["full_sq_log"] = np.log1p(train["full_sq"])
# test["full_sq_log"] = np.log1p(test["full_sq"])


train["life_sq"] = predict_missing_variable(train.drop(["id","timestamp"],axis=1),"life_sq")
train["full_sq"] = predict_missing_variable(train.drop(["id","timestamp"],axis=1),"full_sq")
train["floor"] = predict_missing_variable(train.drop(["id","timestamp"],axis=1),"floor")


def transform_labels(x_train, x_test):

    # for c in x_train.columns:
    #     if x_train[c].dtype == 'object':
    #         x_train[c].fillna("",inplace = True)
    #         x_test[c].fillna("", inplace=True)
    #
    #         lb = LabelBinarizer()
    #         lb.fit(list(x_train[c].values) + list(x_test[c].values))
    #         xtrn = lb.transform(list(x_train[c].values))
    #         xtst = lb.transform(list(x_test[c].values))
    #
    #         x_train = pd.DataFrame(pd.concat([x_train,pd.DataFrame(xtrn,columns=[c + "_" + str(i)for i in range(np.shape(xtrn)[1])])],axis=1))
    #         x_test = pd.DataFrame(pd.concat([x_test, pd.DataFrame(xtst,columns=[c + "_" + str(i)for i in range(np.shape(xtst)[1])])], axis=1))
    #
    #         x_train.drop(c,axis=1,inplace=True)
    #         x_test.drop(c,axis=1,inplace=True)
    #
    # return x_train, x_test


    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values))
            x_train[c] = lbl.transform(list(x_train[c].values))
            # x_train.drop(c,axis=1,inplace=True)

    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_test[c].values))
            x_test[c] = lbl.transform(list(x_test[c].values))
            # x_test.drop(c,axis=1,inplace=True)
    return x_train, x_test


y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)


x_train, x_test = transform_labels(x_train, x_test)

predictions = []

xgb_params = {

    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50)
#cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
predictions.append(y_predict)

predictions = np.mean(predictions,axis=0)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

current_date = datetime.datetime.now()
output.to_csv('outputs/xgbSub{0}-{1}-{2}-{3}.csv'.format(current_date.day,current_date.hour,current_date.minute,current_date.second), index=False)