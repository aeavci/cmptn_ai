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

url = "C:/Users/user/cmptn_ai/data/car data.csv"
data = pd.read_csv(url)
data.drop(labels='Car_Name', axis=1, inplace=True)
data = pd.get_dummies(data=data, drop_first=True)
X = data.drop(["Selling_Price"], axis=1)
y = data["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7)

regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print("accuracy: " + str(regr.score(X_test, y_test)*100) + "%")

dec_tree_regressor = DecisionTreeRegressor()
dec_tree_regressor.fit(X_train, y_train)
dec_pred = dec_tree_regressor.predict(X_test)
print("accuracy: " + str(dec_tree_regressor.score(X_test, y_test)*100) + "%")

knnregressor = KNeighborsRegressor()
knnregressor.fit(X_train, y_train)
knn_pred = knnregressor.predict(X_test)
print("accuracy: " + str(knnregressor.score(X_test, y_test)*100) + "%")

las_reg = Lasso()
las_reg.fit(X_train, y_train)
y_pred = las_reg.predict(X_test)
print("accuracy: " + str(las_reg.score(X_test, y_test)*100) + "%")