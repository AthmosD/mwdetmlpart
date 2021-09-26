import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

benign=pd.read_csv('benign_ngrams.txt',header=None)
malware=pd.read_csv('malware_ngrams.txt',header=None)

benign.to_csv('benign.csv')
malware.to_csv('malware.csv')

f_benign=pd.read_csv('clean_benign.csv',index_col=None)
f_malware=pd.read_csv('clean_malware.csv',index_col=None)

benign_malware=pd.concat([f_malware,f_benign],axis=0)
benign_malware= benign_malware.sample(frac=1).reset_index(drop=True)
benign_malware.to_csv('final_dataset.csv',index=False)

dataset=pd.read_csv('final_dataset.csv')
dataset.drop(['ID'],axis=1,inplace=True)
X=dataset.iloc[ : , :-1]
y=dataset['Target']

kf = KFold(n_splits=5)
regressor = RandomForestRegressor(max_depth=2, random_state=0)
for train_index, test_index in kf.split(X):

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    regressor.fit(x_train,y_train)
    prediction = regressor.predict(x_test)

    print('mse',mean_squared_error(y_test, prediction, squared=True))
    print('rmse',mean_squared_error(y_test, prediction, squared=False))
    print('r^2',r2_score(y_test, prediction))


