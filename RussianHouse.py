import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class RussianHouse:

    def __init__(self):
        pass

    def addFeatures(self, df):

        # df["timestamp"] = pd.to_datetime(df["timestamp"])
        # df["year"] = df["timestamp"].dt.year

        #df = df.loc[df["year"] >= 2014]

        # df["state"].fillna(int(np.mean(df["state"])))
        # df["material"].fillna(int(np.mean(df["material"])))
        # df["full_sq"].fillna(np.mean(df["full_sq"]))
        # df["life_sq"].fillna(np.mean(df["life_sq"]))
        #
        # df["floor_diff"] = df["max_floor"] - df["floor"]
        # df["living_square_diff"] = df["full_sq"] - df["life_sq"]
        # df["livingwithoutkitchen_square_diff"] = df["life_sq"] - df["kitch_sq"]
        pass



    def addComplexFeatures(self, train, test, featureName):
        #train[featureName].fillna("", inplace = True)
        #test[featureName].fillna("", inplace = True)
        lb = preprocessing.LabelEncoder()
        lb.fit(list(train[featureName].values))
        train[featureName] = lb.transform(list(train[featureName].values))

        lb = preprocessing.LabelEncoder()
        lb.fit(list(test[featureName].values))
        test[featureName] = lb.transform(list(test[featureName].values))

    def transform(self, train, test):



        self.addFeatures(train)
        self.addFeatures(test)

        train.drop(["id", "timestamp"], axis=1, inplace=True)
        test.drop(["id", "timestamp"], axis=1, inplace=True)

        for x in train.columns:
            if train[x].dtype == 'object':
                self.addComplexFeatures(train,test,x)

        for x in train.columns:
            if train[x].dtypes == "object":
                train.drop(x, axis=1, inplace=True)

        for x in test.columns:
            if test[x].dtypes == "object":
                test.drop(x, axis=1, inplace=True)

        return train,test

    def corr_plot(self, dataframe, top_n, target, fig_x, fig_y):
        train = dataframe.copy()

        corrmat = dataframe.corr()
        # top_n - top n correlations +1 since price is included
        top_n = top_n + 1
        cols = corrmat.nlargest(top_n, target)[target].index
        cm = np.corrcoef(train[cols].values.T)
        f, ax = plt.subplots(figsize=(fig_x, fig_y))
        sns.set(font_scale=1.25)
        cmap = plt.cm.viridis
        hm = sns.heatmap(cm, cbar=False, annot=True, square=True, cmap=cmap, fmt='.2f', annot_kws={'size': 10},
                         yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        return cols




