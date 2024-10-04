import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import talib
from sklearn.metrics import accuracy_score, precision_score
import joblib
from xgboost import XGBClassifier

startdate = datetime.datetime(2000, 1, 1)
enddate = datetime.datetime(2024, 1, 1)
ticker = 'JPM'

data = yf.download(ticker, start=startdate, end=enddate, interval='1d')
data.drop('Adj Close', axis=1, inplace=True)

data['Return'] = data['Close'].pct_change(1)
data['ReturnSign'] = np.where(data['Return'] < 0, 0, np.where(data['Return'] == 0, 1, 2))

# Calculate indicators and shift them
data['VWAP'] = (data['Open'] * data['Volume']).shift(1)
data['Rsi'] = talib.RSI(data['Close'], timeperiod=10).shift(1)
data['mom'] = talib.MOM(data['Close'], timeperiod=10).shift(1)
data['Adx'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=10).shift(1)
data['EMA7'] = talib.EMA(data['Close'], timeperiod=7).shift(1)
data['EMA12'] = talib.EMA(data['Close'], timeperiod=12).shift(1)
data['EMA100'] = talib.EMA(data['Close'], timeperiod=100).shift(1)
data['EMA200'] = talib.EMA(data['Close'], timeperiod=200).shift(1)
_, _, data['MACDH'] = talib.MACD(data['Close'])
data['BBLow'], data['BBMid'], data['BBHigh'] = talib.BBANDS(data['Close'], timeperiod=11)
data['BBLow'] = data['BBLow'].shift(1)
data['BBMid'] = data['BBMid'].shift(1)
data['BBHigh'] = data['BBHigh'].shift(1)

# Shifted features for different periods (3, 7, 20, 50)
shifts = [3, 7, 20, 50]
for shift in shifts:
    data[f'VWAP_{shift}'] = data['VWAP'].shift(shift)
    data[f'Rsi_{shift}'] = data['Rsi'].shift(shift)
    data[f'mom_{shift}'] = data['mom'].shift(shift)
    data[f'Adx_{shift}'] = data['Adx'].shift(shift)
    data[f'EMA7_{shift}'] = data['EMA7'].shift(shift)
    data[f'EMA12_{shift}'] = data['EMA12'].shift(shift)
    data[f'EMA100_{shift}'] = data['EMA100'].shift(shift)
    data[f'EMA200_{shift}'] = data['EMA200'].shift(shift)
    data[f'BBLow_{shift}'] = data['BBLow'].shift(shift)
    data[f'BBMid_{shift}'] = data['BBMid'].shift(shift)
    data[f'BBHigh_{shift}'] = data['BBHigh'].shift(shift)

data = data.dropna(axis=0)
print(data)

x = data.drop(['Return', 'ReturnSign', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
y = data['ReturnSign']

split = int(len(data) * 0.8)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]

reg = XGBClassifier()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

score_acc = accuracy_score(y_test, y_pred)
score_pre = precision_score(y_test, y_pred, average='weighted')
print('The accuracy and precision scores are:')
print(score_acc)
print(score_pre)

joblib.dump(reg, 'stock_prediction_model_605.pkl')
loaded_model = joblib.load('stock_prediction_model_605.pkl')
