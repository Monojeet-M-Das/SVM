import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import yfinance as yf
from sklearn.model_selection import ParameterGrid

def download_data(stock, start, end):
    data = {}
    ticker = yf.download(stock, start, end)
    data['Close'] = ticker['Close'].squeeze()
    return pd.DataFrame(data)

def construct_features(data, lags=2):

    # calculate the lagged closing prices (name=close)
    for i in range(0, lags):
        data['Lag%s'%str(i+1)] = data['Close'].shift(i+1)
        
    # calculate the percent actual change
    data['Today Change'] = data['Close']
    data['Today Change'] = data['Today Change'].pct_change() * 100

    # calculate the lags in percentage (normalization)
    for i in range(0, lags):
        data['Lag%s'%str(i+1)] = data['Lag%s'%str(i+1)].pct_change() * 100

    # direction - the target variable
    data['Direction'] = np.where(data['Today Change'] > 0, 1, -1)

if __name__ == '__main__':
    start = datetime.datetime(2017,1,1)
    end = datetime.datetime(2018,1,1)

    stock_data = download_data('^GSPC', start, end)
    construct_features(stock_data)
    stock_data.dropna(inplace=True)

    X = stock_data[['Lag1', 'Lag2']]
    y = stock_data['Direction']

    # 80% for training - 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # possible values for the parameters
    parameters =  {'gamma': [0.01, 0.001, 0.0001, 0.00001, 0.000001],
                   'C': [1, 10, 100, 1000, 10000, 100000]}
    
    grid = list(ParameterGrid(parameters))

    best_accuracy = 0
    best_parameter = None

    for p in grid:
        model = SVC(C=p['C'], gamma=p['gamma'])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print('Accurace of the model: %.2f' % accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameter = p

    print('We have found the best parameters...')
    print(best_accuracy)
    print(best_parameter)