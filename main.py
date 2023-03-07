import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tfl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import os
import stat

def stockPredictionModel(Company_name, Number_of_share, Buying_date, Selling_date):
    df = pd.read_csv('Nifty50.csv')
    df.head()
    df["Date"] = pd.to_datetime(df["Date"])
    # name_of_stock = input("Enter Name Of Stock")
    # number_of_share = int(input("Number Of Share"))
    Y = df[df["Symbol"] == Company_name][["Close"]]
    # start_date = input("Start Date")
    # end_date = input("End Date")
    Buying_date = pd.to_datetime(Buying_date)
    Selling_date = pd.to_datetime(Selling_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    Y = scaler.fit_transform(np.array(Y).reshape(-1, 1))

    training_size = int(len(Y) * 0.65)
    test_size = len(Y) - training_size
    train_data, test_data = Y[0:training_size, :], Y[training_size:len(Y), :1]

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    ### Create the Stacked LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=5, batch_size=64, verbose=1)

    last_date = pd.to_datetime("2023-2-15")
    final_output_1 = []
    final_output_2 = []

    Buy_date = Buying_date - last_date
    Sell_date = Selling_date - last_date
    if int(Buy_date.days) > 0:
        n_steps = 100
        x_input = test_data[len(test_data) - n_steps:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        i = 0
        while (i < int(Buy_date.days)):
            if (len(temp_input) > 100):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                #                 print("{} day input {}".format(i,x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                #                 print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                final_output_1.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                #                 print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                #                 print(len(temp_input))
                final_output_1.extend(yhat.tolist())
                i = i + 1

    if int(Sell_date.days) > 0:
        n_steps = 100
        x_input = test_data[len(test_data) - n_steps:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        i = 0
        while (i < int(Sell_date.days)):
            if (len(temp_input) > 100):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                #                 print("{} day input {}".format(i,x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                #                 print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                final_output_2.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                #                 print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                #                 print(len(temp_input))
                final_output_2.extend(yhat.tolist())
                i = i + 1
    if int(Sell_date.days) > 0 and int(Buy_date.days) > 0:
        invests = scaler.inverse_transform(final_output_1)[-1]
        returns = scaler.inverse_transform(final_output_2)[-1]
        Final_Amount = ((returns - invests) * (int(Number_of_share)))
    elif int(Sell_date.days) > 0 and int(Buy_date.days) < 0:
        invests = (df[(df['Symbol'] == Company_name) & (df['Date'] == Buying_date)]['Close'].values[0]).reshape(1, -1)
        invests = invests[-1] * (int(Number_of_share))
        returns = (scaler.inverse_transform(final_output_2)[-1]) * (int(Number_of_share))
        Final_Amount = (returns - invests)
    else:
        invests = (df[(df['Symbol'] == Company_name) & (df['Date'] == Buying_date)]['Close'].values[0]).reshape(1, -1)
        invests = invests[-1] * (int(Number_of_share))
        returns = (df[(df['Symbol'] == Company_name) & (df['Date'] == Selling_date)]['Close'].values[0]).reshape(1, -1)
        returns = returns[-1] * (int(Number_of_share))

        Final_Amount = (returns - invests)

    # In[2]:

    if Final_Amount[0] > 0:
        # print("Profit: ", Final_Amount[0])
        return Final_Amount[0]
    else:
        # print("Loss: ", Final_Amount[0])
        return Final_Amount[0]


def main():
    # giving a title
    st.title("Stock Price Prediction")

    # getting the input data from user
    result = 0

    Company_name = st.text_input("Name of Company")
    Number_of_share = st.number_input("Number of share")
    Buying_date = st.date_input("Stock buying date")
    Selling_date = st.date_input("Stock selling date")

    if st.button("Predict"):
        result = stockPredictionModel(Company_name, Number_of_share, Buying_date, Selling_date)
        st.success(result)


if __name__ == "__main__":
    main()

