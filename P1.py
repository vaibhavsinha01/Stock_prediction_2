import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score
import joblib

startdate = datetime.datetime(2000,1,1)
enddate = datetime.datetime(2023,1,1)
ticker = 'JPM'

data = yf.download(ticker,start=startdate,end=enddate,interval='1d')
data.drop('Adj Close',axis=1,inplace=True)

data['Return'] = data['Close'].pct_change(1)
data['ReturnSign'] = np.sign(data['Return'])
data['VWAP'] = (data['Open']*data['Volume']).shift(1) # volume
data['Rsi'] = (talib.RSI(data['Close'],timeperiod=10)).shift(1) # momentum
data['mom'] = (talib.MOM(data['Close'],timeperiod=10)).shift(1) # momentum
data['Adx'] = (talib.ADX(data['High'],data['Low'],data['Close'],timeperiod=10)).shift(1) # volatility
data['EMA7'] = (talib.EMA(data['Close'],timeperiod=7)).shift(1) # trend
data['EMA12'] = (talib.EMA(data['Close'],timeperiod=12)).shift(1) # trend
data['EMA100'] = (talib.EMA(data['Close'],timeperiod=100)).shift(1) # trend
data['EMA200'] = (talib.EMA(data['Close'],timeperiod=200)).shift(1) # trend
_,_,data['MACDH'] = talib.MACD(data['Close'])
data['BBLow'],data['BBMid'],data['BBHigh'] = (talib.BBANDS(data['Close'],timeperiod=11)) # bands
data['BBLow'] = data['BBLow'].shift(1)
data['BBMid'] = data['BBMid'].shift(1)
data['BBHigh'] = data['BBHigh'].shift(1)
data = data.dropna(axis=0)
print(data)

x = data[['VWAP','Rsi','mom','Adx','EMA7','EMA12','EMA100','EMA200','BBLow','BBMid','BBHigh','MACDH']]
y = data['ReturnSign']

split = int(len(data)*0.8)

scaler = StandardScaler()
scaler.fit_transform(x)
# now make train and test for x,y each and use standard scaler then calculate accuracy score

reg = RandomForestClassifier()

x_train = x.iloc[:split]
y_train = y.iloc[:split]
x_test = x.iloc[split:]
y_test = y.iloc[split:]

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

score_acc = accuracy_score(y_test,y_pred)
score_pre = precision_score(y_test,y_pred,average='weighted')
print('The accuracy,precision,recall scores are:')
print(score_acc)
print(score_pre)

joblib.dump(reg,'stock_prediction_model.pkl')
loaded_model = joblib.load('stock_prediction_model.pkl')