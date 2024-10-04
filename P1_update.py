import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score
import joblib
from xgboost import XGBClassifier

startdate = datetime.datetime(2000,1,1)
enddate = datetime.datetime(2024,1,1)
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

data['VWAP_3'] = (data['Open']*data['Volume']).shift(3) # volume
data['Rsi_3'] = (talib.RSI(data['Close'],timeperiod=10)).shift(3) # momentum
data['mom_3'] = (talib.MOM(data['Close'],timeperiod=10)).shift(3) # momentum
data['Adx_3'] = (talib.ADX(data['High'],data['Low'],data['Close'],timeperiod=10)).shift(3) # volatility
data['EMA7_3'] = (talib.EMA(data['Close'],timeperiod=7)).shift(3) # trend
data['EMA12_3'] = (talib.EMA(data['Close'],timeperiod=12)).shift(3) # trend
data['EMA100_3'] = (talib.EMA(data['Close'],timeperiod=100)).shift(3) # trend
data['EMA200_3'] = (talib.EMA(data['Close'],timeperiod=200)).shift(3) # trend
data['BBLow_3'],data['BBMid_3'],data['BBHigh_3'] = (talib.BBANDS(data['Close'],timeperiod=11)) # bands
data['BBLow_3'] = data['BBLow_3'].shift(3)
data['BBMid_3'] = data['BBMid_3'].shift(3)
data['BBHigh_3'] = data['BBHigh_3'].shift(3)

data['VWAP_7'] = (data['Open']*data['Volume']).shift(7) # volume
data['Rsi_7'] = (talib.RSI(data['Close'],timeperiod=10)).shift(7) # momentum
data['mom_7'] = (talib.MOM(data['Close'],timeperiod=10)).shift(7) # momentum
data['Adx_7'] = (talib.ADX(data['High'],data['Low'],data['Close'],timeperiod=10)).shift(7) # volatility
data['EMA7_7'] = (talib.EMA(data['Close'],timeperiod=7)).shift(7) # trend
data['EMA12_7'] = (talib.EMA(data['Close'],timeperiod=12)).shift(7) # trend
data['EMA100_7'] = (talib.EMA(data['Close'],timeperiod=100)).shift(7) # trend
data['EMA200_7'] = (talib.EMA(data['Close'],timeperiod=200)).shift(7) # trend
data['BBLow_7'],data['BBMid_7'],data['BBHigh_7'] = (talib.BBANDS(data['Close'],timeperiod=11)) # bands
data['BBLow_7'] = data['BBLow_7'].shift(7)
data['BBMid_7'] = data['BBMid_7'].shift(7)
data['BBHigh_7'] = data['BBHigh_7'].shift(7)

data['VWAP_20'] = (data['Open']*data['Volume']).shift(20) # volume
data['Rsi_20'] = (talib.RSI(data['Close'],timeperiod=10)).shift(20) # momentum
data['mom_20'] = (talib.MOM(data['Close'],timeperiod=10)).shift(20) # momentum
data['Adx_20'] = (talib.ADX(data['High'],data['Low'],data['Close'],timeperiod=10)).shift(20) # volatility
data['EMA7_20'] = (talib.EMA(data['Close'],timeperiod=7)).shift(20) # trend
data['EMA12_20'] = (talib.EMA(data['Close'],timeperiod=12)).shift(20) # trend
data['EMA100_20'] = (talib.EMA(data['Close'],timeperiod=100)).shift(20) # trend
data['EMA200_20'] = (talib.EMA(data['Close'],timeperiod=200)).shift(20) # trend
data['BBLow_20'],data['BBMid_20'],data['BBHigh_20'] = (talib.BBANDS(data['Close'],timeperiod=11)) # bands
data['BBLow_20'] = data['BBLow_20'].shift(20)
data['BBMid_20'] = data['BBMid_20'].shift(20)
data['BBHigh_20'] = data['BBHigh_20'].shift(20)

data['VWAP_50'] = (data['Open']*data['Volume']).shift(50) # volume
data['Rsi_50'] = (talib.RSI(data['Close'],timeperiod=10)).shift(50) # momentum
data['mom_50'] = (talib.MOM(data['Close'],timeperiod=10)).shift(50) # momentum
data['Adx_50'] = (talib.ADX(data['High'],data['Low'],data['Close'],timeperiod=10)).shift(50) # volatility
data['EMA7_50'] = (talib.EMA(data['Close'],timeperiod=7)).shift(50) # trend
data['EMA12_50'] = (talib.EMA(data['Close'],timeperiod=12)).shift(50) # trend
data['EMA100_50'] = (talib.EMA(data['Close'],timeperiod=100)).shift(50) # trend
data['EMA200_50'] = (talib.EMA(data['Close'],timeperiod=200)).shift(50) # trend
data['BBLow_50'],data['BBMid_50'],data['BBHigh_50'] = (talib.BBANDS(data['Close'],timeperiod=11)) # bands
data['BBLow_50'] = data['BBLow_50'].shift(50)
data['BBMid_50'] = data['BBMid_50'].shift(50)
data['BBHigh_50'] = data['BBHigh_50'].shift(50)

data = data.dropna(axis=0)
print(data)

x = data[['VWAP','Rsi','mom','Adx','EMA7','EMA12','EMA100','EMA200','BBLow','BBMid','BBHigh','MACDH','VWAP_3','Rsi_3','mom_3','Adx_3','EMA7_3','EMA12_3','EMA100_3','EMA200_3','BBLow_3','BBMid_3','BBHigh_3','VWAP_7','Rsi_7','mom_7','Adx_7','EMA7_7','EMA12_7','EMA100_7','EMA200_7','BBLow_7','BBMid_7','BBHigh_7','VWAP_20','Rsi_20','mom_20','Adx_20','EMA7_20','EMA12_20','EMA100_20','EMA200_20','BBLow_20','BBMid_20','BBHigh_20','VWAP_50','Rsi_50','mom_50','Adx_50','EMA7_50','EMA12_50','EMA100_50','EMA200_50','BBLow_50','BBMid_50','BBHigh_50']]
y = data['ReturnSign']

split = int(len(data)*0.8)

scaler = StandardScaler()
scaler.fit_transform(x)
# now make train and test for x,y each and use standard scaler then calculate accuracy score

reg = RandomForestClassifier()
# reg = XGBClassifier()


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