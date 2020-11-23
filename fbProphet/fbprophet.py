# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:13:25 2020

@author: wangjingxian
"""

#cd python/fbprophet


from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df=dataframe =pd.read_csv('E:/data_mining/Time_series_prediction/dataset/our-dataset.csv')
m=Prophet()
m = Prophet(growth='logistic',changepoint_prior_scale=2)
df['cap']=10000
#df=dataframe =pd.read_csv('E:/data_mining/Time_series_prediction/prophet-master/python/fbprophet/tests/data.csv')
#df=dataframe =pd.read_csv('E:/data_mining/Time_series_prediction/dataset/our-dataset.csv')
m.fit(df)
#future=m.make_future_dataframe(periods=365)
future=m.make_future_dataframe(freq='M',periods=12)
#future=m.make_future_dataframe(periods=12)
future['cap']=10000
result=m.predict(future)
#result.to_csv('E:/data_mining/Time_series_prediction/result.csv')
result.to_csv('E:/data_mining/Time_series_prediction/result/result_ourdataset.csv')

m.plot(result)
plt.show()

test=result[['ds','yhat']].tail(60) 
print(result[-20:])





#包含修改参数的源码
import numpy as np
import pandas as pd
import pystan
from fbprophet import Prophet
import matplotlib.pyplot as plt

pdata = pd.read_csv("data/20200301-nCoV-hb.csv")
pdata.rename(columns={'date':'ds','confirmed':'y'},inplace=True)
pdata['ds'] = pd.to_datetime(pdata['ds'],format='%Y%m%d')
m = Prophet(growth='logistic',changepoint_prior_scale=2,changepoints=['2020-01-24','2020-02-14','2020-02-20'])
pdata['cap']=69000
m.fit(pdata)
#创建一个包含预测时间的dataframe
future = m.make_future_dataframe(periods=20)
future['cap']=69000
forecast = m.predict(future)
m.plot(forecast)
plt.show()
test = forecast[['ds','yhat']].tail(20)
print(test[-20:])








from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df=dataframe =pd.read_csv('E:/data_mining/Time_series_prediction/dataset/our-dataset.csv')
m = Prophet(growth='linear',changepoint_prior_scale=4,seasonality_prior_scale=20)
m = Prophet(growth='linear',changepoint_prior_scale=1,seasonality_prior_scale=20)
m = Prophet(growth='linear',changepoint_prior_scale=6,seasonality_prior_scale=15)
m = Prophet(growth='linear',changepoint_prior_scale=3,seasonality_prior_scale=30)
m = Prophet(growth='linear',changepoint_prior_scale=8,seasonality_prior_scale=5)
m = Prophet(growth='linear',changepoint_prior_scale=8)
m = Prophet()
m = Prophet(changepoint_prior_scale=1)
m = Prophet(changepoint_prior_scale=0.3)
#m = Prophet(growth='linear',changepoint_prior_scale=3,seasonality_prior_scale=15,interval_width=1.6)
m = Prophet()
m.fit(df)
#future=m.make_future_dataframe(periods=365)
#future=m.make_future_dataframe(freq='M',periods=12,mcmc_samples=1,interval_width=1.6)
future=m.make_future_dataframe(freq='M',periods=12)
#future=m.make_future_dataframe(pd.period_range('2017-12', freq='M', periods=12))
#future=m.make_future_dataframe(periods=12)

result=m.predict(future)
#result.to_csv('E:/data_mining/Time_series_prediction/result.csv')
result.to_csv('E:/data_mining/Time_series_prediction/result/result_ourdataset5.csv')

m.plot(result)
plt.show()

test=result[['ds','yhat']].tail(60) 
print(result[-20:])







