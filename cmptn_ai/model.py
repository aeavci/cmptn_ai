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
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



data = None
label_data = None
X = None
y = None


def myFunc(z, t):
    """Help users to predict the price of diamonds, cars and abalone.
    """
    if(t == 1):
        """Diamonds Dataset"""
        url = "C:/Users/avci/cmptn_ai/cmptn_ai/data/diamonds.csv"
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
        """Car Price Prediction Dataset"""
        url = "C:/Users/avci/cmptn_ai/cmptn_ai/data/car data.csv"
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
        """Abalone Dataset"""
        url = "C:/Users/avci/cmptn_ai/cmptn_ai/data/abalone.csv"
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
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        y_pred = lin_reg.predict(X_test)
        print("accuracy: " + str(lin_reg.score(X_test, y_test)*100) + "%")
        m = str(lin_reg.score(X_test, y_test)*100) + "%"
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(lin_reg, file)
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
    elif(z == 4):
        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)
        rfr_pred = rfr.predict(X_test)
        print("accuracy: " + str(rfr.score(X_test, y_test)*100) + "%")
        m = str(rfr.score(X_test, y_test)*100) + "%"
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(rfr, file)
        return m
    elif(z == 5):
        gbr = GradientBoostingRegressor()
        gbr.fit(X_train, y_train)
        gbr_pred = gbr.predict(X_test)
        print("accuracy: " + str(gbr.score(X_test, y_test)*100) + "%")
        m = str(gbr.score(X_test, y_test)*100) + "%"
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(gbr, file)
        return m
    elif(z == 6):
        svr = SVR()
        svr.fit(X_train, y_train)
        svr_pred = svr.predict(X_test)
        print("accuracy: " + str(svr.score(X_test, y_test)*100) + "%")
        m = str(svr.score(X_test, y_test)*100) + "%"
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(svr, file)
        return m
    elif(z == 7):
        ridge = Ridge()
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
        print("accuracy: " + str(ridge.score(X_test, y_test)*100) + "%")
        m = str(ridge.score(X_test, y_test)*100) + "%"
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(ridge, file)
        return m
    elif(z==8):
        elasticnet = ElasticNet()
        elasticnet.fit(X_train, y_train)
        elasticnet_pred = elasticnet.predict(X_test)
        print("accuracy: " + str(elasticnet.score(X_test, y_test)*100) + "%")
        m = str(elasticnet.score(X_test, y_test)*100) + "%"
        file = open('comp.pkl', 'wb')
        # dump information to that file
        pickle.dump(elasticnet, file)
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
