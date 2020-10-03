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
    
# Before we move on to the next step we will seperate the dataset into train, validation, test split


# Function from https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
def data_split(examples, labels, train_frac, random_state=5):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        examples, labels, train_size=train_frac, random_state=random_state)

    X_val, X_test, Y_val, Y_test = train_test_split(X_tmp,
                                                    Y_tmp,
                                                    train_size=0.5,
                                                    random_state=random_state)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

tree = DecisionTreeRegressor()
tree.fit(X_train, Y_train)
tree_predictios = tree.predict(X_val)
print("Validation")
scores(Y_val, tree_predictios)

# print("\nTest")
# tree_predictions_test = tree.predict(X_test)
# scores(Y_test, tree_predictions_test)
    
