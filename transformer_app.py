
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

st.title("Traffic Volume Forecasting with Transformer and ARIMA")
uploaded_file = st.file_uploader("Upload a CSV file with 'Timestamp' and 'Traffic Volume' columns", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

df1 = pd.read_csv(uploaded_file)
df1['Timestamp'] = pd.to_datetime(df1['Timestamp'])
df1.set_index('Timestamp', inplace=True)

vehicles = df1['Traffic Volume'].fillna(method='ffill')
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df1[['Traffic Volume']])

# Stationarity test
result = adfuller(vehicles)
st.write(f"ADF Statistic: {result[0]}")
st.write(f"p-value: {result[1]}")

vehicles_diff = vehicles.diff().dropna() if result[1] > 0.05 else vehicles

# ACF and PACF plots
st.subheader("ACF and PACF Plots")
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(vehicles_diff, lags=40, ax=ax[0])
plot_pacf(vehicles_diff, lags=40, ax=ax[1])
st.pyplot(fig)

# ARIMA Forecast
model_arima = ARIMA(vehicles, order=(2,1,2))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=30)
st.subheader("ARIMA Forecast")
fig_arima, ax_arima = plt.subplots()
ax_arima.plot(vehicles[-100:], label="Actual")
ax_arima.plot(forecast_arima, label="ARIMA Forecast")
ax_arima.legend()
st.pyplot(fig_arima)

# Create sequences
SEQ_LEN = 24
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

# Transformer model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    return LayerNormalization(epsilon=1e-6)(x + x_ff)

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

input_shape = (X.shape[1], X.shape[2])
model_transformer = build_transformer_model(
    input_shape, head_size=128, num_heads=4, ff_dim=128, num_transformer_blocks=2, mlp_units=[64, 32],
    dropout=0.1, mlp_dropout=0.1
)
model_transformer.compile(loss="mse", optimizer="adam")
model_transformer.fit(X, y, epochs=20, batch_size=16, verbose=1)

# Evaluate Transformer
y_pred_transformer = model_transformer.predict(X)
y_actual_inv = scaler.inverse_transform(y.reshape(-1, 1))
y_pred_transformer_inv = scaler.inverse_transform(y_pred_transformer)

mae_transformer = mean_absolute_error(y_actual_inv, y_pred_transformer_inv)
mse_transformer = mean_squared_error(y_actual_inv, y_pred_transformer_inv)
rmse_transformer = np.sqrt(mse_transformer)
r2_transformer = r2_score(y_actual_inv, y_pred_transformer_inv)

st.subheader("Transformer Model Evaluation")
st.write(f"MAE: {mae_transformer:.4f}")
st.write(f"MSE: {mse_transformer:.4f}")
st.write(f"RMSE: {rmse_transformer:.4f}")
st.write(f"R²: {r2_transformer:.4f}")

fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
ax_pred.plot(y_actual_inv, label='Actual')
ax_pred.plot(y_pred_transformer_inv, label='Transformer Prediction')
ax_pred.set_title('Actual vs. Transformer Predicted Traffic Volume')
ax_pred.set_xlabel('Time Step')
ax_pred.set_ylabel('Traffic Volume')
ax_pred.legend()
st.pyplot(fig_pred)
