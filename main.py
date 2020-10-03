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
    
    ''' need to set X to columns
    set y to sneaker
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #splitting into 70% training 30% test
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #can use to test accuracy
    '''
    
