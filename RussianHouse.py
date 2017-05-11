import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class RussianHouse:

    def __init__(self):
        pass

    def addFeatures(self, df):

        # df["timestamp"] = pd.to_datetime(df["timestamp"])
        # df["year"] = df["timestamp"].dt.year
        # df["month"] = df["timestamp"].dt.month
        # df["day"] = df["timestamp"].dt.day


        # df["state"].fillna(int(np.mean(df["state"])))
        # df["material"].fillna(int(np.mean(df["material"])))
        # df["full_sq"].fillna(np.mean(df["full_sq"]))
        # df["life_sq"].fillna(np.mean(df["life_sq"]))

        # df["floor_diff"] = df["max_floor"] - df["floor"]
        # df["living_square_diff"] = df["full_sq"] - df["life_sq"]
        # df["livingwithoutkitchen_square_diff"] = df["life_sq"] - df["kitch_sq"]
        # df["square_per_room"] = df["life_sq"] / df["num_room"]
        # df["year_diff"] = 2016 - df["year"]
        # df["full_sq_log"] = np.log1p(df["full_sq"])
        # df["floor_log"] = np.log1p(df["floor"])
        # df["life_sq_log"] = np.log1p(df["life_sq"])
        # df["full_sq_log"] = np.log1p(df["full_sq"])
        # df["num_room_log"] = np.log1p(df["num_room"])

        pass



    def reduceDimension(self, train, test):

        imp = Imputer(strategy="mean", axis=0)
        train = imp.fit_transform(train)
        test= imp.fit_transform(test)

        pca = PCA(n_components=50)
        pca.fit(train)
        train = pca.transform(train)

        pca = PCA(n_components=50)
        pca.fit(test)
        test = pca.transform(test)


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

        # train["life_sq"].dropna(inplace=True)
        # train["num_room"].dropna(inplace=True)
        # train["material"].dropna(inplace=True)
        # train["price_doc"] = np.log1p(train["price_doc"].values)
        # price = train["price_doc"]
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

        # train.drop(["price_doc"],inplace=True,axis=1)
        # self.reduceDimension(train,test)
        # train["price_doc"] = price


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




