# !wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
# !tar -xzvf ta-lib-0.4.0-src.tar.gz
# %cd ta-lib
# !./configure --prefix=/usr
# !make
# !make install
# !pip install Ta-Lib

# !pip install yfinance

import yfinance as yf
import pandas as pd
from talib import abstract
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree

stk = yf.Ticker('SPY')
# 取得 2000 年至今的資料
data1 = stk.history(start = '2000-01-01')
# 簡化資料，只取開、高、低、收以及成交量
data = data1[['Open', 'High', 'Low', 'Close','Volume']]
data.columns = ['open','high','low','close','volume']

ta_list = ['MACD','RSI','MOM','STOCH','ADX','WILLR','OBV','CCI','MA','CORREL','STDDEV','VAR','CMO']
# 快速計算與整理因子
for x in ta_list:
    output = eval('abstract.'+x+'(data)')
    output.name = x.lower() if type(output) == pd.core.series.Series else None
    data = pd.merge(data, pd.DataFrame(output), left_on = data.index, right_on = output.index)
    data = data.set_index('key_0')
    
data = data.dropna()
pred_day=10

#預測漲跌
for i in range(pred_day):
    data[f'trend_day{i+1}'] = np.where(data.close.shift(-(i+1)) > data.close, 1, 0)

# #預預測buy.sell signals
# for i in range(pred_day):
#   data[f'trend_day{i+1}']=0

# print(len(data))

# for i in range(pred_day):
#   for j in range(len(data)-(i+1)):
#     if data['close'][j+i+1]/data['close'][j]>1.01:
#       data[f'trend_day{i+1}'][j]=2
#     elif data['close'][j+i+1]/data['close'][j]<0.99:
#       data[f'trend_day{i+1}'][j]=0
#     else:
#       data[f'trend_day{i+1}'][j]=1

features = data.iloc[:,:-pred_day]
labels = data.iloc[:,-pred_day:]

for i in range(len(labels.columns)):
    # specify the feature set, target set, the test size and random_state to select records randomly
    train_X , test_X , train_y, test_y = train_test_split(features, labels.iloc[:,i], test_size=0.3,random_state=0) 
    scaling = MinMaxScaler(feature_range=(0,1)).fit(train_X)
    train_X = scaling.transform(train_X)
    test_X = scaling.transform(test_X)

    xgbrModel=XGBClassifier(max_depth=28,n_estimators=140,learning_rate=0.2)#max_depth=28,n_estimators=140,learning_rate=0.2
    xgbrModel.fit(train_X,train_y)

    print(f'day{i+1}的Accuracy:{xgbrModel.score(test_X,test_y)}')
    
# # 要計算 AUC 的話，要從 metrics 裡匯入 roc_curve 以及 auc
# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
# def plot_roc_curve(fper, tper):
#     plt.plot(fper, tper, color='red', label='ROC')
#     plt.plot([0, 1], [0, 1], color='green', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic Curve')
#     plt.legend()
#     plt.show()

# # 計算 ROC 曲線
# false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y,prediction)

# plot_roc_curve(false_positive_rate,true_positive_rate)
# # 計算 AUC 面積
# auc(false_positive_rate, true_positive_rate)

# import matplotlib.pyplot as plt

# # 測試一批深度參數，一般而言深度不太會超過 3x，我們這邊示範 1 到 50 好了
# depth_parameters = np.arange(1,50)
# n_parameters=np.arange(1,150)
# learn_parameters=np.arange(0.1,0.3,0.01)
# # 準備兩個容器，一個裝所有參數下的訓練階段 AUC；另一個裝所有參數下的測試階段 AUC
# train_auc= []
# test_auc = []
# # 根據每一個參數跑迴圈
# for test_depth in learn_parameters:
#     # 根據該深度參數，創立一個決策樹模型，取名 temp_model
#     temp_model = XGBClassifier(max_depth=28,n_estimators=140,learning_rate=test_depth)
#     # 讓 temp_model 根據 train 學習樣本進行學習
#     temp_model.fit(train_X, train_y)
#     # 讓學習後的 temp_model 分別根據 train 學習樣本以及 test 測試樣本進行測驗
#     train_predictions = temp_model.predict(train_X)
#     test_predictions = temp_model.predict(test_X)
#     # 計算學習樣本的 AUC，並且紀錄起來
#     false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y, train_predictions)
#     auc_area = auc(false_positive_rate, true_positive_rate)
#     train_auc.append(auc_area)
#     # 計算測試樣本的 AUC，並且紀錄起來
#     false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, test_predictions)
#     auc_area = auc(false_positive_rate, true_positive_rate)
#     test_auc.append(auc_area)

# # 繪圖視覺化
# plt.figure(figsize = (14,10))
# plt.plot(learn_parameters, train_auc, 'b', label = 'Train AUC')
# plt.plot(learn_parameters, test_auc, 'r', label = 'Test AUC')
# plt.ylabel('AUC')
# plt.xlabel('depth parameter')
# plt.show()

# for i in range(len(test_auc)):
#   if test_auc[i]==max(test_auc):
#     print(i)
