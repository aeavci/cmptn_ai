import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
data = None
label_data = None
X = None
y = None


def myFunc(z, t):
    if(t == 1):
        url = "C:/Users/user/cmptn_ai/data/diamonds.csv"
        data = pd.read_csv(url)
        data = data.drop(["Unnamed: 0"], axis=1)
        data = data.drop(data[data["x"] == 0].index)
        data = data.drop(data[data["y"] == 0].index)
        data = data.drop(data[data["z"] == 0].index)

        s = (data.dtypes == "object")
        object_cols = list(s[s].index)
        print("Categorical variables:")
        print(object_cols)

        # Make copy to avoid changing original data
        label_data = data.copy()
        # Apply label encoder to each column with categorical data
        label_encoder = LabelEncoder()
        for col in object_cols:
            label_data[col] = label_encoder.fit_transform(label_data[col])
        print(label_data.head())
        cmap = sns.diverging_palette(70, 20, s=50, l=40, n=6, as_cmap=True)
        corrmat = label_data.corr()
        f, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(corrmat, cmap=cmap, annot=True)
        X = label_data.drop(["price"], axis=1)
        y = label_data["price"]
        # plt.savefig('save_as_a_png.png')
    elif(t == 2):
        url = "C:/Users/user/cmptn_ai/data/car data.csv"
        data = pd.read_csv(url)
        data.drop(labels='Car_Name', axis=1, inplace=True)
        correlations = data.corr()
        indx = correlations.index
        plt.figure(figsize=(26, 22))
        sns.heatmap(data[indx].corr(), annot=True, cmap="YlGnBu")
        plt.savefig('carpricecorr.png')
        data = pd.get_dummies(data=data, drop_first=True)
        X = data.drop(["Selling_Price"], axis=1)
        y = data["Selling_Price"]
    else:
        url = "C:/Users/user/cmptn_ai/data/abalone.csv"
        data = pd.read_csv(url)
        data['age'] = data['Rings']+1.5
        data.drop('Rings', axis=1, inplace=True)
        numerical_features = data.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(20, 7))
        sns.heatmap(data[numerical_features].corr(), annot=True)
        plt.savefig('abalone.png')
        data = pd.get_dummies(data, drop_first=True)
        X = data.drop("age", axis=1)
        y = data["age"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7)
    if(z == 1):
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        print("accuracy: " + str(regr.score(X_test, y_test)*100) + "%")
        m = str(regr.score(X_test, y_test)*100) + "%"
        # open a file, where you ant to store the data
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(regr, file)
        return m
    elif(z == 2):
        dec_tree_regressor = DecisionTreeRegressor()
        dec_tree_regressor.fit(X_train, y_train)
        dec_pred = dec_tree_regressor.predict(X_test)
        print("accuracy: " + str(dec_tree_regressor.score(X_test, y_test)*100) + "%")
        m = str(dec_tree_regressor.score(X_test, y_test)*100) + "%"
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(dec_tree_regressor, file)
        return m
    elif(z == 3):
        knnregressor = KNeighborsRegressor()
        knnregressor.fit(X_train, y_train)
        knn_pred = knnregressor.predict(X_test)
        print("accuracy: " + str(knnregressor.score(X_test, y_test)*100) + "%")
        m = str(knnregressor.score(X_test, y_test)*100) + "%"
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(knnregressor, file)
        return m
    else:
        las_reg = Lasso()
        las_reg.fit(X_train, y_train)
        y_pred = las_reg.predict(X_test)
        print("accuracy: " + str(las_reg.score(X_test, y_test)*100) + "%")
        m = str(las_reg.score(X_test, y_test)*100) + "%"
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(las_reg, file)
        return m


myFunc(1, 3)
