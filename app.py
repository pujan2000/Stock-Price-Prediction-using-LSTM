import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow import keras
import streamlit as st
from datetime import date

plt.style.use('fivethirtyeight')
import yfinance as yf
yf.pdr_override()

st.title("Stock Trend Prediction")

today = date.today()
start = st.text_input("Enter Start Date", "2010-01-01")
end = st.text_input("Enter End Date", today)
user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = wb.get_data_yahoo(user_input, start, end)

#Describing the Data

st.subheader('Data from ' + start[0:4] + " to " + end[0:4])
st.write(df.head())

st.subheader('Summary Statistics')
st.write(df.describe())

#Visualizations

st.subheader("Closing Price Chart")
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

#Moving Averages
def moving_avg(days):
    ma = df.Close.rolling(days).mean()
    return ma

st.subheader("Closing Price Chart with 100 day MA and 200 day MA")
fig = plt.figure(figsize = (12,6))
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

#Splitting the data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df['Close'])*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df['Close'])*0.7):int(len(df['Close']))])
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


#Splitting training data into x_train and y_train

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[(i-100):i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train) ,np.array(y_train)

#Loading ML Model
loaded_model = keras.models.load_model('keras_model.h5')

# Splitting the test data into x_test and y_test
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

#Predictions

y_predicted = loaded_model.predict(x_test)
scale_factor = 1/scaler.scale_
y_predicted = scale_factor*y_predicted
y_test = scale_factor*y_test

#Visualising the Predicted Stock Price with the Actual Stock Price

st.subheader("Stock Price Prediction using LSTM")
fig = plt.figure(figsize=(14,9))
plt.plot(y_test,"r",label = "Original Stock Price")
plt.plot(y_predicted,"b",label = "Predicted Stock Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)
