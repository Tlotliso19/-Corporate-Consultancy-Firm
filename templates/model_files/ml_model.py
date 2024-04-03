import pickle
import numpy as np
import pandas as  pd 
import matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#from code import df_scaler,data_spliter
%run code.ipynb
from sklearn.metrics import precision_score
#from sklearn.externals import joblib
path=r'C:\Users\makoboshane\Downloads\archive (4)\original.csv'
data_df=df_scaler(path)
x_train, x_test, y_train, y_test=data_spliter(data_df)
clf = tree.DecisionTreeClassifier(max_depth=4)
clt=clf.fit(x_train,y_train)
y_predict=clt.predict(x_train)