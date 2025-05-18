import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import keras_tuner as kt

st.title("Traffic Volume Forecasting with ARIMA & GRU")
st.write("Upload a CSV file with a 'Timestamp' column and a 'Traffic Volume' column.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    vehicles = df['Traffic Volume'].fillna(method='ffill')

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Traffic Volume']])

    result = adfuller(vehicles)
    st.write(f"ADF Statistic: {result[0]}")
    st.write(f"p-value: {result[1]}")

    vehicles_diff = vehicles.diff().dropna() if result[1] > 0.05 else vehicles

    st.subheader("ACF and PACF Plots")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(vehicles_diff, lags=40, ax=ax[0])
    plot_pacf(vehicles_diff, lags=40, ax=ax[1])
    st.pyplot(fig)

    st.subheader("ARIMA Model Forecast")
    model_arima = ARIMA(vehicles, order=(2, 1, 2))
    model_arima_fit = model_arima.fit()
    st.text(model_arima_fit.summary())
    forecast_arima = model_arima_fit.forecast(steps=30)

    fig2, ax2 = plt.subplots()
    ax2.plot(vehicles[-100:], label="Actual")
    ax2.plot(forecast_arima, label="ARIMA Forecast")
    ax2.legend()
    st.pyplot(fig2)

    SEQ_LEN = 24
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, SEQ_LEN)

    model_gru = Sequential([
        GRU(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(32, activation='relu'),
        Dense(y.shape[1])
    ])
    model_gru.compile(optimizer='adam', loss='mse')
    model_gru.fit(X, y, epochs=20, batch_size=16, verbose=0)

    y_pred_gru = model_gru.predict(X)
    y_actual_inv = scaler.inverse_transform(y)
    y_pred_gru_inv = scaler.inverse_transform(y_pred_gru)

    st.subheader("GRU Model Evaluation")
    st.write(f"MAE: {mean_absolute_error(y_actual_inv, y_pred_gru_inv):.4f}")
    st.write(f"MSE: {mean_squared_error(y_actual_inv, y_pred_gru_inv):.4f}")
    st.write(f"R²: {r2_score(y_actual_inv, y_pred_gru_inv):.4f}")

    fig3, ax3 = plt.subplots()
    ax3.plot(y_actual_inv[:, 0], label='Actual')
    ax3.plot(y_pred_gru_inv[:, 0], label='GRU Predicted')
    ax3.set_title('GRU: Actual vs Predicted')
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("Tuning GRU Model")
    def build_gru_model(hp):
        model = Sequential()
        model.add(GRU(
            units=hp.Int('units', 32, 128, step=32),
            input_shape=(X.shape[1], X.shape[2]),
            return_sequences=hp.Boolean('return_sequences')
        ))
        if hp.Boolean('return_sequences'):
            model.add(GRU(units=hp.Int('units_2', 32, 128, step=32)))
        model.add(Dense(hp.Int('dense_units', 16, 64, step=16), activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']), loss='mse')
        return model

    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    tuner = kt.RandomSearch(
        build_gru_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='gru_tuning',
        project_name='traffic_volume_gru'
    )

    with st.spinner("Tuning..."):
        tuner.search(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), verbose=0)
    best_hps = tuner.get_best_hyperparameters(1)[0]

    st.write("Best Hyperparameters:")
    st.write(f"Units: {best_hps.get('units')}")
    st.write(f"Return Sequences: {best_hps.get('return_sequences')}")
    st.write(f"Units 2: {best_hps.get('units_2') if best_hps.get('return_sequences') else 'N/A'}")
    st.write(f"Dense Units: {best_hps.get('dense_units')}")
    st.write(f"Optimizer: {best_hps.get('optimizer')}")

    best_model = tuner.get_best_models(1)[0]
    best_model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    y_pred_best_gru = best_model.predict(X)

    st.subheader("Tuned GRU Model Evaluation")
    st.write(f"MAE: {mean_absolute_error(y, y_pred_best_gru):.4f}")
    st.write(f"MSE: {mean_squared_error(y, y_pred_best_gru):.4f}")
    st.write(f"R²: {r2_score(y, y_pred_best_gru):.4f}")

    fig4, ax4 = plt.subplots()
    ax4.plot(scaler.inverse_transform(y.reshape(-1, 1)), label='Actual')
    ax4.plot(scaler.inverse_transform(y_pred_best_gru), label='Tuned GRU Predicted')
    ax4.set_title('Tuned GRU: Actual vs Predicted')
    ax4.legend()
    st.pyplot(fig4)
else:
    st.warning("Please upload a CSV file to proceed.")