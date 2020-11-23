# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:01:02 2020

@author: wangjingxian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import trange
import numpy

data_csv = pd.read_excel('E:/data_mining/Time_series_prediction/dataset/work_order_quantity.xls', usecols=[1])
#plt.plot(data_csv)
data_csv=data_csv[0:60]


#data_prediction = data_prediction.values
#data_prediction = data_prediction.astype('float32')


# 数据预处理
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: (x-min_value) / scalar, dataset))


data_prediction=dataset[48:60]


#dataset = list(map(lambda x: x / scalar, dataset))
def create_dataset(dataset, look_back=12):
    dataX=[]
    dataY=[]
    for i in range(len(dataset) - look_back):
    #for i in range(len(dataset)):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
        #dataY.append(dataset[i])
    return np.array(dataX), np.array(dataY)


data_prediction=np.array(data_prediction)


#dataset=list(data_csv.values)
# 创建好输入输出
data_X, data_Y = create_dataset(dataset)

#print(data_X)
#print(len(data_X))
#print(data_Y)
#print(len(data_Y))



# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.75)    #48个月的数据用于训练，剩下12个月的数据用于测试
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
#print(train_X)
#print(len(train_X))
train_Y = data_Y[:train_size]
#print(train_Y)
#print(len(train_Y))


test_X = data_X[train_size:]
test_Y = data_Y[train_size:]


#print(train_X)
#print(len(train_X))
train_X = train_X.reshape(-1, 1, 12)
#print(train_X)
#print(len(train_X))
train_Y = train_Y.reshape(-1, 1, 1)
#print(train_Y)
#print(len(train_Y))


test_X = test_X.reshape(-1, 1, 12)
test_Y = test_Y.reshape(-1, 1, 1)
#print(test_X)
#print(test_Y)
#print(len(test_X))
#print(len(test_Y))

data_prediction=data_prediction.reshape(-1, 1, 12)

data_prediction=torch.from_numpy(data_prediction)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
#print(train_x)
#print(train_y)

test_x = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_Y)
#print(test_x)
#print(test_y)


# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=4):
        super(lstm_reg, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.reg = nn.Linear(hidden_size, output_size) # 回归
        
    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


#net = lstm_reg(12, 4)
net = lstm_reg(12,6)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)


# 开始训练
for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0: # 每 100 次输出结果
        #print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))


net = net.eval() # 转换成测试模式
data_X = data_X.reshape(-1, 1, 12)
#dataset=create_test(dataset)
#data_X = dataset.reshape(-1, 1, 12)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data) # 测试集的预测结果

data_prediction=net(data_prediction)
print(data_prediction)
data_prediction = data_prediction.view(-1).data.numpy()
data_prediction = list(map(lambda x: int(x*scalar+min_value), data_prediction))
print(data_prediction)



#print(pred_test)
#print(len(pred_test))
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()

'''
# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')
'''

pred_test1 = list(map(lambda x: int(x*scalar+min_value), pred_test))
print(pred_test1)
print(len(pred_test1))


data_csv=np.array(data_csv[12:60])
data_csv=data_csv.T
print(data_csv[0])
print(len(data_csv[0]))


# 画出实际结果和预测的结果
plt.plot(pred_test1, 'r', label='prediction1')
plt.plot(data_csv[0], 'g', label='real1')
plt.legend(loc='best')







def make_forecast(model: net, look_back_buffer: numpy.ndarray, timesteps: int=1, batch_size: int=1):
    forecast_predict = numpy.empty((0, 1), dtype=numpy.float32)
    #print(forecast_predict)
    for _ in trange(timesteps, desc='predicting data\t', mininterval=1.0):
        # make prediction with current lookback buffer
        cur_predict = model(look_back_buffer, batch_size)
        #print(cur_predict)
        # add prediction to result
        forecast_predict = numpy.concatenate([forecast_predict, cur_predict], axis=0)
        #print(forecast_predict)
        # add new axis to prediction to make it suitable as input
        cur_predict = numpy.reshape(cur_predict, (cur_predict.shape[1], cur_predict.shape[0], 1))
        # remove oldest prediction from buffer
        look_back_buffer = numpy.delete(look_back_buffer, 0, axis=1)
        # concat buffer with newest prediction
        look_back_buffer = numpy.concatenate([look_back_buffer, cur_predict], axis=1)
    return forecast_predict

test_x = test_x.reshape(1, 12)
forecast_predict=forecast_predict = make_forecast(net, test_x[-1::], timesteps=12, batch_size=1)
print(forecast_predict)





'''
test_X = test_X.reshape(-1, 1, 12)
#dataset=create_test(dataset)
#data_X = dataset.reshape(-1, 1, 12)
test_X = torch.from_numpy(test_X)
var_data = Variable(test_X)
pred_test = net(var_data) # 测试集的预测结果

#print(pred_test)
#print(len(pred_test))
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()

pred_test = list(map(lambda x: int(x*scalar+min_value), pred_test))
print(pred_test)
'''


'''
train_X = train_X.reshape(-1, 1, 12)
#dataset=create_test(dataset)
#data_X = dataset.reshape(-1, 1, 12)
train_X = torch.from_numpy(train_X)
var_data = Variable(train_X)
pred_train = net(var_data) # 测试集的预测结果

#print(pred_test)
#print(len(pred_test))
# 改变输出的格式
pred_train = pred_train.view(-1).data.numpy()

pred_train = list(map(lambda x: int(x*scalar+min_value), pred_train))
print(pred_train)
'''



'''
data = pd.read_excel('E:/data_mining/Time_series_prediction/dataset/work_order_quantity.xls', usecols=[1])
data_prediction=data[48:60]
print(data_prediction)
# 数据预处理
data_prediction = data_prediction.dropna()
data_prediction = data_prediction.values
data_prediction = data_prediction.astype('float32')
max_value = np.max(data_prediction)
min_value = np.min(data_prediction)
scalar = max_value - min_value
data_prediction = list(map(lambda x: (x-min_value) / scalar, data_prediction))



def create_dataset(dataset, look_back=12):
    dataX=[]
    for i in range(len(dataset) - look_back):
    #for i in range(len(dataset)):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        #dataY.append(dataset[i])
    return np.array(dataX)



data_prediction = create_dataset(data_prediction)

data_prediction = data_prediction.reshape(-1, 1, 12)
data_prediction = torch.from_numpy(data_prediction)

print(data_prediction)

#var_data = Variable(data_prediction)

#net = net.eval() # 转换成测试模式

pred = net.predict(var_data)

print(pred)

pred= list(map(lambda x: int(x*scalar+min_value), pred))
print(pred)

'''