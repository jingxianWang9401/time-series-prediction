# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:03:35 2020

@author: wangjingxian
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


data=pd.read_excel('E:/data_mining/Time_series_prediction/succeed_model/Holt-Winters/work_order_quantity.xls')
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


#Holt-Winters季节性预测模型
from statsmodels.tsa.api import ExponentialSmoothing
#调参数
#fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=12, trend='add', seasonal='add', ).fit()
#fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=12, trend='mul', seasonal='add', ).fit()
#fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=12, trend='add', seasonal='mul', ).fit()
#fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=12, trend='mul', seasonal='mul', ).fit()
#fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=12, trend='mul', seasonal='mul',damped=True ).fit()
#一部分训练数据进行训练，剩余验证数据进行验证
fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=12, trend='mul', seasonal='mul',damped=True).fit()

y_hat_avg = test.copy()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
print(y_hat_avg['Holt_Winter'])

plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
#plt.legend(loc='best')
#plt.show()



#所有的已知数据均用于训练，再向后预测一年的数据
train1=data[0:60]
test1=data[60:]
print(train1)
print(test1)
fit2 = ExponentialSmoothing(np.asarray(train1['Count']), seasonal_periods=12, trend='mul', seasonal='mul',damped=True).fit()
y_hat_avg1 = test1.copy()
y_hat_avg1['Holt_Winter'] = fit2.forecast(len(test1))
print(y_hat_avg1['Holt_Winter'])


plt.figure(figsize=(16, 8))
plt.plot(train1['Count'], label='Train1')
plt.plot(test1['Count'], label='Test1')
plt.plot(y_hat_avg1['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


