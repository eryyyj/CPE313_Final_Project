# lstm_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

st.title("Traffic Volume Forecasting with LSTM and ARIMA")

# File uploader
uploaded_file = st.file_uploader("Upload your traffic CSV file", type=["csv"])
if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Load and preprocess data
df = pd.read_csv(uploaded_file)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
vehicles = df['Traffic Volume'].fillna(method='ffill')

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Traffic Volume']])

# ADF Test
result = adfuller(vehicles)
st.subheader("ADF Test")
st.write(f"ADF Statistic: {result[0]}")
st.write(f"p-value: {result[1]}")
vehicles_diff = vehicles.diff().dropna() if result[1] > 0.05 else vehicles

# ACF & PACF Plots
st.subheader("ACF and PACF Plots")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(vehicles_diff, lags=40, ax=axes[0])
plot_pacf(vehicles_diff, lags=40, ax=axes[1])
st.pyplot(fig)

# ARIMA Forecasting
st.subheader("ARIMA Forecasting")
model = ARIMA(vehicles, order=(2, 1, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

fig2, ax2 = plt.subplots()
ax2.plot(vehicles[-100:], label="Actual")
ax2.plot(forecast.index, forecast, label="Forecast")
ax2.legend()
st.pyplot(fig2)

# LSTM Sequence Creation
SEQ_LEN = 24
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

# Basic LSTM Model
st.subheader("LSTM Model Training")
model_dl = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(32, activation='relu'),
    Dense(1)
])
model_dl.compile(optimizer='adam', loss='mse')
model_dl.fit(X, y, epochs=20, batch_size=16, verbose=0)

y_pred = model_dl.predict(X)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

st.write(f"LSTM MAE: {mae:.4f}")
st.write(f"LSTM RMSE: {rmse:.4f}")
st.write(f"LSTM R²: {r2:.4f}")

# Optimized LSTM
st.subheader("Optimized LSTM Model")
model_dl_opt = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)
])
model_dl_opt.compile(optimizer='adam', loss='mse')
history = model_dl_opt.fit(X, y, epochs=30, batch_size=16, verbose=0)

y_pred_opt = model_dl_opt.predict(X)
mae_opt = mean_absolute_error(y, y_pred_opt)
rmse_opt = np.sqrt(mean_squared_error(y, y_pred_opt))
r2_opt = r2_score(y, y_pred_opt)

st.write(f"Optimized LSTM MAE: {mae_opt:.4f}")
st.write(f"Optimized LSTM RMSE: {rmse_opt:.4f}")
st.write(f"Optimized LSTM R²: {r2_opt:.4f}")

fig3, ax3 = plt.subplots()
ax3.plot(history.history['loss'])
ax3.set_title('Training Loss (Optimized LSTM)')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
st.pyplot(fig3)