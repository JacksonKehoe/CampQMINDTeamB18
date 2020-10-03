import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


file = 'finalPredict.csv'
df = pd.read_csv(file)

print(len(df.columns))
print(df.head())

def standardize(data): 
    scaler = StandardScaler()
    
    return scaler.fit_transform(data)
    
