import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import array
import urllib
import os
import pandas as pd
import datetime
from tkinter import *
import tkinter as tk
from tkinter import messagebox
import math
import numpy
import requests
from time import sleep, strftime
import random
from random import randint
from bs4 import BeautifulSoup
from selenium import webdriver
import shutil
import alpaca_trade_api as tradeapi
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Flatten



print('\n')

def close_window():
    global entry
    entry = E.get()
    root.destroy()


zz=True   
while(zz==True):  
    root = Tk()
    E = tk.Entry(root)
    E.pack(anchor = CENTER)
    L1 = Label(root, text="¿Que acción quieres predecir el dia de hoy? (NASDAQ) ")
    L1.pack( side = LEFT)
    B = Button(root, text = "OK", command = close_window)
    B.pack(anchor = S,side = LEFT)
    root.mainloop()



    entry=entry.lower()

    if (entry =='apple') or (entry=='aapl'):
        accion='AAPL.csv'
        break
    elif (entry== 'google') or (entry=='goog'):
        accion='GOOG.csv'
        break
    elif (entry == 'msft') or (entry=='microsoft'):
        accion='MSFT.csv'
        break
    elif (entry == 'tesla') or (entry=='tsla'):
        accion='TSLA.csv'
        break
    else:
        zz=messagebox.askretrycancel(message="Lo siento esta acción aún no conózco como predecirla. Hasta el momento solo se predecir las siguientes acciones: \n-AAPL \n-GOOG \n-MSFT \n-TSLA", 
                                        title="Oops!!!!!!!!")

    if zz==False:
        break




url = 'https://www.marketbeat.com/stocks/NASDAQ/'+accion.split('.')[0]+'/price-target/?MostRecent=0'

response = requests.get(url)
# print(bool(response))
soup = BeautifulSoup(response.text, "html.parser")
tabla1=soup.find_all("tbody")
filas_ = tabla1[1].find_all("tr")
filas = tabla1[1].find_all("td")
lista_reco=[]
for i in range(len(filas)):
    x=filas[i].text
    lista_reco.append(x)
del lista_reco[69]
fechas=[]
broker=[]
rating=[]
price_target=[]
for i in range(len(lista_reco)):
    if i %7==0:
        x=lista_reco[i]
        fechas.append(x)
    elif i %7==1:
        y=lista_reco[i]
        broker.append(y)
    elif i %7==3:
        z=lista_reco[i]
        rating.append(z)
    elif i %7==4:
        a=lista_reco[i]
        price_target.append(a)
df_recome = pd.DataFrame({'Date':fechas,'Broker':broker,'Rating':rating,'Price_target':price_target})
df_recome = df_recome.sort_values('Date')
fechas_correctas=[]
for i in range(len(df_recome)):
    d=df_recome["Date"][i]
    d_object=datetime.strptime(d,"%m/%d/%Y")
    fechas_correctas.append(d_object)
df_recome = pd.DataFrame({'Date':fechas_correctas,'Broker':broker,'Rating':rating})#'Price_target':price_target})
df_recome = df_recome.sort_values('Date')
df_recome["Weekday"]=df_recome['Date'].map(lambda x:x.isoweekday())
t1=timedelta(days=1)
fechas_semana=[]
for i in range(len(df_recome)):
    x=df_recome["Weekday"][i]
    if (x==1) or (x==2) or (x==3) or (x==4) or (x==5):
        y=df_recome["Date"][i]
#     elif x==5:
#         y=df_recome["Date"][i]+t1*3
    elif x==6:
        y=df_recome["Date"][i]+t1*2
    else:
        y=df_recome["Date"][i]+t1
    fechas_semana.append(y)
fechas_semana.sort()
df_recome["Fechas_semana"]=fechas_semana
most_fre_rating=df_recome.groupby(['Fechas_semana'])["Rating"].agg(lambda x:x.value_counts().index[0])
most_fre_rating=list(most_fre_rating)
most_brok_rating=df_recome.groupby(['Fechas_semana'])["Broker"].agg(lambda x:x.value_counts().index[0])
most_brok_rating=list(most_brok_rating)
df_final_=df_recome.groupby("Fechas_semana").count()
df_final_["Rating"]=most_fre_rating
df_final_["Broker"]=most_brok_rating
df_final_=df_final_.drop(['Date', 'Weekday'], axis=1)
df_final_=df_final_.reset_index()
lista_freq=list(df_final_["Rating"].unique())
dicc_=dict()
for i in range(len(lista_freq)):
    if ((lista_freq[i]=="Hold") or (lista_freq[i]=="Neutral") or (lista_freq[i]=='Equal Weight') or (lista_freq[i]=='Market Perform ➝ Market Perform')
       or (lista_freq[i]=='Buy ➝ Neutral') or (lista_freq[i]=='Neutral ➝ Neutral') or (lista_freq[i]=='Overweight ➝ Neutral') or 
       (lista_freq[i]=='Outperform ➝ Market Perform') or (lista_freq[i]=='Mkt Perform ➝ Market Perform') or (lista_freq[i]=='Equal Weight ➝ Equal Weight')
       or (lista_freq[i]=='Buy ➝ Hold') or (lista_freq[i]=='Outperform ➝ In-Line') or (lista_freq[i]=='Hold ➝ Hold') or 
       (lista_freq[i]=='Positive ➝ Hold') or (lista_freq[i]=='Neutral ➝ Hold') or  (lista_freq[i]=='In-Line')):
        dicc_[lista_freq[i]]=0
    elif ((lista_freq[i]=='Buy') or (lista_freq[i]=='Overweight') or (lista_freq[i]=='Outperform ➝ Buy') or (lista_freq[i]=='Outperform')
          or (lista_freq[i]=='Sector Weight ➝ Overweight') or (lista_freq[i]=='In-Line ➝ Buy') or (lista_freq[i]=='Strong-Buy') or
         (lista_freq[i]=='Buy ➝ Buy') or (lista_freq[i]=='Overweight ➝ Buy') or (lista_freq[i]=='Overweight ➝ Overweight') or
         (lista_freq[i]=='Hold ➝ Buy') or (lista_freq[i]=='Average ➝ Buy') or (lista_freq[i]== 'Neutral ➝ Buy' ) or (lista_freq[i]=='') or
         (lista_freq[i]=='Outperform ➝ Outperform') or (lista_freq[i]=='Positive ➝ Outperform') or (lista_freq[i]=='Positive ➝ Buy') or
         (lista_freq[i]=='Buy AAPL') or (lista_freq[i]=='Market Perform ➝ Outperform') or  (lista_freq[i]=='Positive ➝ Overweight') or
         (lista_freq[i]=='Sell ➝ Outperform') or (lista_freq[i]=='Equal Weight ➝ Overweight') or (lista_freq[i]=='Outperform ➝ Overweight') or
         (lista_freq[i]=='Reduce ➝ Buy') or (lista_freq[i]=='Top Pick ➝ Above Average') or (lista_freq[i]=='Buy ➝ Positive') or (lista_freq[i]=='Market Perform') or
         (lista_freq[i]=='Buy ➝ Strong-Buy') or (lista_freq[i]=='Positive') or (lista_freq[i]=='Neutral ➝ Overweight') or (lista_freq[i]=='Buy ➝ $107.97') or
         (lista_freq[i]=='Neutral ➝ Outperform') or (lista_freq[i]=='Buy ➝ In-Line') or (lista_freq[i]=='Buy ➝ Outperform') or (lista_freq[i]=='Buy ➝ Focus List') or
         (lista_freq[i]=='Strong-Buy ➝ Strong-Buy') ):
        dicc_[lista_freq[i]]=1
    elif ((lista_freq[i]=='Sell') or (lista_freq[i]=='Hold ➝ Reduce') or (lista_freq[i]=='Neutral ➝ Sell') or (lista_freq[i]=='Hold ➝ Sell') or 
         (lista_freq[i]=='Neutral ➝ Underweight') or (lista_freq[i]=='Underperform') or (lista_freq[i]=='$1,195.88')  or (lista_freq[i]=='$1,186.96') or 
         (lista_freq[i]=='$1,205.50') or (lista_freq[i]=='$105.12') or (lista_freq[i]=='$104.40') or (lista_freq[i]=='Underperform ➝ Underperform')):
        dicc_[lista_freq[i]]=-1      
df_final_["Rating"]=df_final_["Rating"].replace(dicc_)
print('\n')
print('Recomendaciones de Brokers durante los últimos 3 años: \n1 Comprar \n0 Mantener \n-1 Vender')
print('\n')
print(df_final_["Rating"].value_counts())
print(' \n')
filename1 = datetime.now().strftime("%Y-%m-%d")
df_final_.to_csv(f"Webscrapping {accion.split('.')[0]} {filename1}.csv", index=False)
ddf=pd.read_csv(f"Webscrapping {accion.split('.')[0]} {filename1}.csv")
'''***************************************************************'''
# Personalizacion de la accion

df_stocks_=pd.read_csv(accion) 
del df_stocks_["Open"]
del df_stocks_["High"]
del df_stocks_["Low"]
del df_stocks_["Adj Close"]
del df_stocks_["Volume"]
close_values=df_stocks_["Close"].values
percentage_change=[]
for i in range(1,len(close_values)):
    x=((close_values[i]-close_values[i-1])/close_values[i-1]*100)
    percentage_change.append(x)
df_stocks_=df_stocks_.drop([0],axis=0)
df_stocks_["Change_close"]=percentage_change
fechas_correctas1=[]
for i in range(1,len(df_stocks_)+1):
    d1=df_stocks_["Date"][i]
    d_object1=datetime.strptime(d1,"%Y-%m-%d")
    fechas_correctas1.append(d_object1)
df_stocks_["Date"] = fechas_correctas1

# MERGE
df_merge1=pd.merge(left=df_stocks_, right=df_final_, left_on="Date", right_on="Fechas_semana",how="right")
df_merge1=df_merge1.drop(["Fechas_semana"], axis="columns")
reo='Recomendaciones de los ultimos 10 dias de '+accion.split('.')[0]
print(reo)
print(df_merge1.tail(10))



# 3 años de datos 2018   2020
df=pd.read_csv(accion)
columnas=['High', 'Low', 'Adj Close']
df1=df.reset_index()[columnas[0]] # High
df2=df.reset_index()[columnas[1]] # Low
df3=df.reset_index()[columnas[2]] # Adj Close
scaler=MinMaxScaler(feature_range=(0,1))
# Se normalizan los datos
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
df2=scaler.fit_transform(np.array(df2).reshape(-1,1))
df3=scaler.fit_transform(np.array(df3).reshape(-1,1))

##splitting dataset into train and test split
training_size_high=int(len(df1)*0.70)
test_size_high=len(df1)-training_size_high
train_data_high,test_data_high=df1[0:training_size_high,:],df1[training_size_high:len(df1),:1]

training_size_low=int(len(df2)*0.70)
test_size_low=len(df2)-training_size_low
train_data_low,test_data_low=df2[0:training_size_low,:],df2[training_size_low:len(df2),:1]

training_size_close=int(len(df3)*0.70)
test_size_close=len(df3)-training_size_close
train_data_close,test_data_close=df3[0:training_size_close,:],df3[training_size_close:len(df3),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 5
X_train_high, y_train_high = create_dataset(train_data_high, time_step)
X_test_high, ytest_high = create_dataset(test_data_high, time_step)

X_train_low, y_train_low = create_dataset(train_data_low, time_step)
X_test_low, ytest_low = create_dataset(test_data_low, time_step)

X_train_close, y_train_close = create_dataset(train_data_close, time_step)
X_test_close, ytest_close = create_dataset(test_data_close, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train_high =X_train_high.reshape(X_train_high.shape[0],X_train_high.shape[1] , 1)
X_test_high = X_test_high.reshape(X_test_high.shape[0],X_test_high.shape[1] , 1)
X_train_low =X_train_low.reshape(X_train_low.shape[0],X_train_low.shape[1] , 1)
X_test_low = X_test_low.reshape(X_test_low.shape[0],X_test_low.shape[1] , 1)
X_train_close =X_train_close.reshape(X_train_close.shape[0],X_train_close.shape[1] , 1)
X_test_close = X_test_close.reshape(X_test_close.shape[0],X_test_close.shape[1] , 1)

# Modelo High
model=Sequential()
model.add(LSTM(10,return_sequences=True,input_shape=(5,1)))
# model.add(LSTM(50,return_sequences=True))
model.add(LSTM(10))
model.add(Dense(1))


model.compile(loss='mean_squared_error',optimizer='adam')
print('\n')
model.summary()
history_high=model.fit(X_train_high,y_train_high,validation_data=(X_test_high,ytest_high),epochs=40,verbose=0)

### Lets Do the prediction and check performance metrics
train_predict_high=model.predict(X_train_high)
test_predict_high=model.predict(X_test_high)
train_predict_high=scaler.inverse_transform(train_predict_high)
test_predict_high=scaler.inverse_transform(test_predict_high)

### Calculate RMSE performance metrics
print("RMSE_high_train "+str(math.sqrt(mean_squared_error(y_train_high,train_predict_high))))
print("RMSE_high_test "+str(math.sqrt(mean_squared_error(ytest_high,test_predict_high))))

print("*******************************************************************************************")
# Posible error de shape REVISAR len(test_data - los dias que voy a predecir)
x_input_high=test_data_high[146:].reshape(1,-1)
temp_input_high=list(x_input_high)
temp_input_high=temp_input_high[0].tolist()

# demonstrate prediction for next days
lst_output_high=[]
n_steps=5
i=0
while(i<5):
    
    if(len(temp_input_high)>5):
#         print(temp_input)
        x_input_high=np.array(temp_input_high[1:])
#         print("{} day input {}".format(i,x_input_high))
        x_input_high=x_input_high.reshape(1,-1)
        x_input_high = x_input_high.reshape((1, n_steps, 1))
        #print(x_input)
        yhat_high = model.predict(x_input_high, verbose=0)
#         print("{} day output {}".format(i,yhat_high))
        temp_input_high.extend(yhat_high[0].tolist())
        temp_input_high=temp_input_high[1:]
        #print(temp_input)
        lst_output_high.extend(yhat_high.tolist())
        i=i+1
    else:
        x_input_high = x_input_high.reshape((1, n_steps,1))
        yhat_high = model.predict(x_input_high, verbose=0)
#         print(yhat_high[0])
        temp_input_high.extend(yhat_high[0].tolist())
#         print(len(temp_input_high))
        lst_output_high.extend(yhat_high.tolist())
        i=i+1

'''***************************'''
day_new_high=np.arange(1,11)
day_pred_high=np.arange(11,16)
plt.subplot(2,2,1)
plt.title("Predicciones próximos 5 dias - Máximos")
plt.plot(day_new_high,scaler.inverse_transform(df1[len(df1)-len(day_new_high):]))
# plt.show
plt.plot(day_pred_high,scaler.inverse_transform(lst_output_high))
# plt.show()

plt.subplot(2,2,2) 
df_high=df1.tolist()
df_high.extend(lst_output_high)
df_high=scaler.inverse_transform(df_high).tolist()
plt.title("Predicciones próximos 5 dias - Máximos")
plt.plot(df_high[480:])
plt.show()


'''*************************'''

# Modelo Low
model=Sequential()
model.add(LSTM(10,return_sequences=True,input_shape=(5,1)))
# model.add(LSTM(50,return_sequences=True))
model.add(LSTM(10))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')
history_low=model.fit(X_train_low,y_train_low,validation_data=(X_test_low,ytest_low),epochs=40,verbose=0)


### Lets Do the prediction and check performance metrics
train_predict_low=model.predict(X_train_low)
test_predict_low=model.predict(X_test_low)
train_predict_low=scaler.inverse_transform(train_predict_low)
test_predict_low=scaler.inverse_transform(test_predict_low)

### Calculate RMSE performance metrics
print("RMSE_low_train "+str(math.sqrt(mean_squared_error(y_train_low,train_predict_low))))
print("RMSE_low_test "+str(math.sqrt(mean_squared_error(ytest_low,test_predict_low))))

print("*******************************************************************************************")

# Posible error de shape REVISAR len(test_data - los dias que voy a predecir)
x_input_low=test_data_low[146:].reshape(1,-1)
temp_input_low=list(x_input_low)
temp_input_low=temp_input_low[0].tolist()

# demonstrate prediction for next days
lst_output_low=[]
n_steps=5
i=0
while(i<5):
    
    if(len(temp_input_low)>5):
#         print(temp_input)
        x_input_low=np.array(temp_input_low[1:])
#         print("{} day input {}".format(i,x_input_high))
        x_input_low=x_input_low.reshape(1,-1)
        x_input_low= x_input_low.reshape((1, n_steps, 1))
        #print(x_input)
        yhat_low = model.predict(x_input_low, verbose=0)
#         print("{} day output {}".format(i,yhat_high))
        temp_input_low.extend(yhat_low[0].tolist())
        temp_input_low=temp_input_low[1:]
        #print(temp_input)
        lst_output_low.extend(yhat_low.tolist())
        i=i+1
    else:
        x_input_low = x_input_low.reshape((1, n_steps,1))
        yhat_low = model.predict(x_input_low, verbose=0)
#         print(yhat_high[0])
        temp_input_low.extend(yhat_low[0].tolist())
#         print(len(temp_input_high))
        lst_output_low.extend(yhat_low.tolist())
        i=i+1

'''***************************'''
day_new_low=np.arange(1,11)
day_pred_low=np.arange(11,16)
plt.subplot(2,2,1)
plt.title("Predicciones próximos 5 dias - Mínimos ")
plt.plot(day_new_low,scaler.inverse_transform(df2[len(df2)-len(day_new_low):]))
# plt.show
plt.plot(day_pred_low,scaler.inverse_transform(lst_output_low))
# plt.show()

plt.subplot(2,2,2)
df_low=df2.tolist()
df_low.extend(lst_output_low)
df_low=scaler.inverse_transform(df_low).tolist()
plt.title("Predicciones próximos 5 dias - Mínimos ")
plt.plot(df_low[480:])
plt.show()


'''*************************'''



# Modelo Close
model=Sequential()
model.add(LSTM(20,return_sequences=True,input_shape=(5,1)))
# model.add(LSTM(50,return_sequences=True))
model.add(LSTM(20))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

history_close=model.fit(X_train_close,y_train_close,validation_data=(X_test_close,ytest_close),epochs=80,verbose=0)

### Lets Do the prediction and check performance metrics
train_predict_close=model.predict(X_train_close)
test_predict_close=model.predict(X_test_close)
train_predict_close=scaler.inverse_transform(train_predict_close)
test_predict_close=scaler.inverse_transform(test_predict_close)

### Calculate RMSE performance metrics
print("RMSE_close_train "+str(math.sqrt(mean_squared_error(y_train_close,train_predict_close))))
print("RMSE_close_test "+str(math.sqrt(mean_squared_error(ytest_close,test_predict_close))))



# Posible error de shape REVISAR len(test_data - los dias que voy a predecir)
x_input_close=test_data_close[146:].reshape(1,-1)
temp_input_close=list(x_input_close)
temp_input_close=temp_input_close[0].tolist()

# demonstrate prediction for next days
lst_output_close=[]
n_steps=5
i=0
while(i<5):
    
    if(len(temp_input_close)>5):
#         print(temp_input)
        x_input_close=np.array(temp_input_close[1:])
#         print("{} day input {}".format(i,x_input_high))
        x_input_close=x_input_close.reshape(1,-1)
        x_input_close= x_input_close.reshape((1, n_steps, 1))
        #print(x_input)
        yhat_close = model.predict(x_input_close, verbose=0)
#         print("{} day output {}".format(i,yhat_high))
        temp_input_close.extend(yhat_close[0].tolist())
        temp_input_close=temp_input_close[1:]
        #print(temp_input)
        lst_output_close.extend(yhat_close.tolist())
        i=i+1
    else:
        x_input_close= x_input_close.reshape((1, n_steps,1))
        yhat_close = model.predict(x_input_close, verbose=0)
#         print(yhat_high[0])
        temp_input_close.extend(yhat_close[0].tolist())
#         print(len(temp_input_high))
        lst_output_close.extend(yhat_close.tolist())
        i=i+1

'''***************************'''
day_new_close=np.arange(1,11)
day_pred_close=np.arange(11,16)
plt.subplot(2,2,1)
plt.title("Predicciones próximos 5 dias - Cierre Ajustado ")
plt.plot(day_new_close,scaler.inverse_transform(df3[len(df3)-len(day_new_close):]))
# plt.show
plt.plot(day_pred_close,scaler.inverse_transform(lst_output_close))
# plt.show()

plt.subplot(2,2,2)
df_close=df3.tolist()
df_close.extend(lst_output_close)
df_close=scaler.inverse_transform(df_close).tolist()
plt.title("Predicciones próximos 5 dias - Cierra Ajustado ")
plt.plot(df_close[480:])
plt.show()


'''*************************'''


def plot_metrics(history):
    
    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    
# print("High")
plot_metrics(history_high)
# print("Low")
plot_metrics(history_low)
# print("Close")
plot_metrics(history_close)


### Plotting 
# shift train predictions for plotting
look_back=5
trainPredictPlot_high = numpy.empty_like(df1)
trainPredictPlot_high[:, :] = np.nan
trainPredictPlot_high[look_back:len(train_predict_high)+look_back, :] = train_predict_high
# shift test predictions for plotting
testPredictPlot_high = numpy.empty_like(df1)
testPredictPlot_high[:, :] = numpy.nan
testPredictPlot_high[len(train_predict_high)+(look_back*2)+1:len(df1)-1, :] = test_predict_high
# plot baseline and predictions
plt.title('Predicciones_High')
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot_high)
plt.plot(testPredictPlot_high)
plt.legend(['Datos reales ', 'Prediccion con datos de entrenamiento ', 'Prediccion con datos de validacion'])
plt.show()

### Plotting 
# shift train predictions for plotting
look_back=5
trainPredictPlot_low = numpy.empty_like(df2)
trainPredictPlot_low[:, :] = np.nan
trainPredictPlot_low[look_back:len(train_predict_low)+look_back, :] = train_predict_low
# shift test predictions for plotting
testPredictPlot_low = numpy.empty_like(df2)
testPredictPlot_low[:, :] = numpy.nan
testPredictPlot_low[len(train_predict_low)+(look_back*2)+1:len(df2)-1, :] = test_predict_low
# plot baseline and predictions
plt.title('Predicciones_Low')
plt.plot(scaler.inverse_transform(df2))
plt.plot(trainPredictPlot_low)
plt.plot(testPredictPlot_low)
plt.legend(['Datos reales ', 'Prediccion con datos de entrenamiento ', 'Prediccion con datos de validacion'])
plt.show()


### Plotting 
# shift train predictions for plotting
look_back=5
trainPredictPlot_close = numpy.empty_like(df3)
trainPredictPlot_close[:, :] = np.nan
trainPredictPlot_close[look_back:len(train_predict_close)+look_back, :] = train_predict_close
# shift test predictions for plotting
testPredictPlot_close = numpy.empty_like(df3)
testPredictPlot_close[:, :] = numpy.nan
testPredictPlot_close[len(train_predict_close)+(look_back*2)+1:len(df3)-1, :] = test_predict_close
# plot baseline and predictions
plt.title('Predicciones_Close')
plt.plot(scaler.inverse_transform(df3))
plt.plot(trainPredictPlot_close)
plt.plot(testPredictPlot_close)
plt.legend(['Datos reales ', 'Prediccion con datos de entrenamiento ', 'Prediccion con datos de validacion'])
plt.show()

last_price=df.iloc[-1]['Adj Close']
puntuacion_=df_merge1.iloc[-1][-1]

if puntuacion_ ==1:
    puntuacion='COMPRAR'
elif puntuacion_ ==0:
    puntuacion='ESPERAR'
elif puntuacion_==-1:
    puntuacion='VENDER'

price_predicted_high=scaler.inverse_transform(lst_output_high)
price_predicted_high=np.reshape(price_predicted_high,5)

price_predicted_low=scaler.inverse_transform(lst_output_low)
price_predicted_low=np.reshape(price_predicted_low,5)

price_predicted_close=scaler.inverse_transform(lst_output_close)
price_predicted_close=np.reshape(price_predicted_close,5)

prom_high=(price_predicted_high[0]+price_predicted_high[1]+price_predicted_high[2]+price_predicted_high[3]+price_predicted_high[4])/5
prom_low=(price_predicted_low[0]+price_predicted_low[1]+price_predicted_low[2]+price_predicted_low[3]+price_predicted_low[4])/5
prom_close=(price_predicted_close[0]+price_predicted_close[1]+price_predicted_close[2]+price_predicted_close[3]+price_predicted_close[4])/5

last_price=round(last_price,2)
prom_high=round(prom_high, 2)
prom_low=round(prom_low, 2)
prom_close=round(prom_close, 2)



text_last_recom=('La última recomendación de los brokers es ' +str(puntuacion)+ '\nEl último precio real de ' + accion.split('.')[0] + ' es ' + str(last_price) +' \nEl promedio de la predicción (5 dias forward) de precios maximos es: '+ str(prom_high) +
'\nEl promedio de la predicción (5 dias forward) de precios mínimos es: ' +  str(prom_low) +  '\nEl promedio de la predicción (5 dias forward) de precios de cierre es: '+ str(prom_close) +'\n¿DESEA PONER UNA ORDEN EL MERCAD0?')

decision=messagebox.askyesno(message=text_last_recom, title="Recomendación del sistema")

if decision==True:
    def close_window():
        global entry
        entry = selected.get()
        window.destroy()

    window = Tk()
    window.title("CONFIRMAR ORDEN")

    window.geometry('300x50')

    selected = IntVar()

    rad1 = Radiobutton(window,text='COMPRAR', value=1, variable=selected)
    rad2 = Radiobutton(window,text='VENDER', value=-1, variable=selected)
    btn = Button(window, text="CONFIRMAR", command=close_window)

    rad1.grid(column=0, row=0)
    rad2.grid(column=1, row=0)
    btn.grid(column=3, row=0)

    window.mainloop()
    
    if selected.get()==0: #-----------------------------------------    CUIDADO CUCHO CON ESE CERO
        side_='buy'
    elif selected.get()==-1:
        side_='sell'


    def close_window__():
        global entry
        entry = selected2.get()
        window.destroy()

    window = Tk()
    window.title("CONFIRMAR CANTIDAD")

    window.geometry('320x280')

    selected2 = IntVar()

    rad1 = Radiobutton(window,text=1, value=1, variable=selected2)
    rad2 = Radiobutton(window,text=2, value=2, variable=selected2)
    rad3 = Radiobutton(window,text=3, value=3, variable=selected2)
    rad4 = Radiobutton(window,text=4, value=4, variable=selected2)
    rad5 = Radiobutton(window,text=5, value=5, variable=selected2)
    rad6 = Radiobutton(window,text=6, value=6, variable=selected2)
    rad7 = Radiobutton(window,text=7, value=7, variable=selected2)
    rad8 = Radiobutton(window,text=8, value=8, variable=selected2)
    rad9 = Radiobutton(window,text=9, value=9, variable=selected2)
    rad10 = Radiobutton(window,text=10, value=10, variable=selected2)


    btn = Button(window, text="CONFIRMAR", command=close_window__)

    rad1.grid(column=1, row=0)
    rad2.grid(column=1, row=1)
    rad3.grid(column=1, row=2)
    rad4.grid(column=1, row=3)
    rad5.grid(column=1, row=4)
    rad6.grid(column=1, row=5)
    rad7.grid(column=1, row=6)
    rad8.grid(column=1, row=7)
    rad9.grid(column=1, row=8)
    rad10.grid(column=1, row=9)

    btn.grid(column=1, row=11)

    window.mainloop()
    qty=selected2.get()

    def close_window_():
        global entry
        entry = selected1.get()
        window.destroy()

    window = Tk()
    window.title("CONFIRMAR ORDEN - Time in Force")
    window.geometry('350x180')
    selected1 = IntVar()
    rad3 = Radiobutton(window,text='GTC - Good til Canceled', value=0, variable=selected1)
    rad4 = Radiobutton(window,text='FOK - Fill or Kill', value=1, variable=selected1)
    rad5 = Radiobutton(window,text='IOC - Immediate or Cancel', value=2, variable=selected1)
    rad6 = Radiobutton(window,text='OPG - At-the-Open', value=3, variable=selected1)
    rad7 = Radiobutton(window,text='CLS - At-the-Close', value=4, variable=selected1)
    btn = Button(window, text="CONFIRMAR", command=close_window_)
    # spin = Spinbox(window, from_=0, to=15, width=5,command=close_window)

    rad3.grid(column=1, row=2)
    rad4.grid(column=1, row=3)
    rad5.grid(column=1, row=4)
    rad6.grid(column=1, row=5) 
    rad7.grid(column=1, row=6)
    btn.grid(column=1, row=7)

    window.mainloop()


    if selected1.get()==0:
        time_in_force_='gtc'
    elif selected1.get()==1:
        time_in_force_='fok'
    elif selected1.get()==2:
        time_in_force_='ioc'
    elif selected1.get()==3:
        time_in_force_='opg'
    elif selected1.get()==5:
        time_in_force_='cls'

symbol_=accion.split('.')[0]
qty=selected2.get()
side_
# type='market'
time_in_force_
# print(symbol_)
# print(qty)
# print(side_)
# print(type)
# print(time_in_force_)

print('\n')

# First, open the API connection
api = tradeapi.REST(
    'PKKHQYI5RRBF8A0L2OYW',
    'dnVZNFD485Bx5Y9SNGeye6QavFMy8F54OuIvzzJ7',
    'https://paper-api.alpaca.markets'
)

# Get account info
account = api.get_account()
# print(account)

# Check our current balance vs. our balance at the last market close
balance_change = float(account.equity) - float(account.last_equity)
print(f'Today\'s portfolio balance change: ${balance_change}')

# Check how much money we can use to open new positions.
print('${} is available as buying power.'.format(account.buying_power))

# Get a list of all active assets.
# active_assets = api.list_assets(status='active')
# print(active_assets)


# api = tradeapi.REST()

# Submit a market order to buy 1 share of Apple at market price
api.submit_order(
    symbol=symbol_,
    qty=qty,
    side=side_,
    type='market',
    time_in_force=time_in_force_
)