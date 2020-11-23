# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:41:23 2020

@author: wangjingxian
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


data=pd.read_excel('E:/data_mining/Time_series_prediction/succeed_model/SARIMAX/work_order_quantity.xls')
#print(data.head())
#print(data.shape)

train=data[0:48]
test=data[48:]
#print(train)
#print(test)




#原始数据图像表示
data['Timestamp'] = pd.to_datetime(data['Datetime'], format='%m-%Y')  # 4位年用Y，2位年用y
data.index = data['Timestamp']
data = data.resample('M').mean() #按天采样，计算均值

train['Timestamp'] = pd.to_datetime(train['Datetime'], format='%m-%Y')
train.index = train['Timestamp']
train = train.resample('M').mean() #
 
test['Timestamp'] = pd.to_datetime(test['Datetime'], format='%m-%Y')
test.index = test['Timestamp']
test = test.resample('M').mean()

matplotlib.rcParams['font.sans-serif']=['SimHei']  #使用指定的汉字字体类型（此处为黑体）
train.Count.plot(figsize=(15,8),title='Month Ridership',fontsize=14)
plt.show()



#自回归移动平均模型（ARIMA）,季节自回归积分滑动平均外生回归模型 
import statsmodels.api as sm


#调参,使用其中一部分数据作为训练，剩余部分作为验证。
#fit1=sm.tsa.statespace.SARIMAX(train.Count,order=(2,1,4),seasonal_order=(0,1,1,12)).fit()
#fit1=sm.tsa.statespace.SARIMAX(train.Count,order=(2,1,0),seasonal_order=(2,1,0,6)).fit()
#fit1=sm.tsa.statespace.SARIMAX(train.Count,order=(0,1,2),seasonal_order=(0,1,2,6)).fit()
#fit1=sm.tsa.statespace.SARIMAX(train.Count,order=(2,1,2),seasonal_order=(2,1,2,6)).fit()
#fit1=sm.tsa.statespace.SARIMAX(data.Count,order=(1,1,0),seasonal_order=(1,1,0,6)).fit()
fit1=sm.tsa.statespace.SARIMAX(train.Count,order=(1,1,0),seasonal_order=(1,1,0,6)).fit()
#y_hat_avg['SARIMA']=fit1.predict(start="01-2017",end="12-2017",dynamic=True)
y_hat_avg=test.copy()
y_hat_avg['SARIMA']=fit1.predict(start="01-2017",end="12-2017",dynamic=True)
print(y_hat_avg['SARIMA'])
plt.figure(figsize=(16,8))
plt.plot(train['Count'],label='Train')
plt.plot(test['Count'],label='Test')
plt.plot(y_hat_avg['SARIMA'],label='SARIMA')
plt.legend(loc='best')
plt.show()




#使用所有已知数据作为训练，预测后面一年的数据
train1=data[0:60]
predict1=data[60:]
fit2=sm.tsa.statespace.SARIMAX(train1.Count,order=(1,1,0),seasonal_order=(1,1,0,6)).fit()
y_hat_avg2=predict1.copy()
y_hat_avg2['SARIMA']=fit2.predict(start="01-2018",end="12-2018",dynamic=True)
print(y_hat_avg2['SARIMA'])
plt.figure(figsize=(16,8))
plt.plot(train1['Count'],label='Train')
plt.plot(predict1['Count'],label='Test')
plt.plot(y_hat_avg2['SARIMA'],label='SARIMA')
plt.legend(loc='best')
plt.show()



steps = 12
#start_time = data.index[-1]
start_time="01-2018"
forecast_ts = fit2.forecast(steps)
fore = pd.DataFrame()
fore['Datetime'] = pd.date_range(start=start_time ,periods=steps, freq='1M')
fore['Count'] = pd.DataFrame(forecast_ts.values)
fore.index = pd.to_datetime(fore['Datetime'])
print(fore)