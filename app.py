import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Needed for date formatting

# -------------------------------
st.set_page_config(page_title="Crypto LSTM Forecast", layout="centered")
st.title("📈 Cryptocurrency Price Forecast (Next 7 Days)")

# -------------------------------
# 1. Select coin
crypto_id = st.selectbox("Select Coin", ['bitcoin', 'ethereum', 'solana'])
vs_currency = 'usd'

# -------------------------------
# 2. Fetch data from CoinGecko API
@st.cache_data
def get_data(crypto_id, days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency={vs_currency}&days={days}"
    res = requests.get(url).json()
    prices = res['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

with st.spinner("⏳ Fetching market data..."):
    df = get_data(crypto_id)
st.success(f"✅ Data fetched: {len(df)} rows")

# -------------------------------
# 3. Prepare data
def prepare_data(df, sequence_length=60):
    if len(df) <= sequence_length:
        st.error(f"❌ Not enough data to train. Need at least {sequence_length + 1} rows, got {len(df)}.")
        return None, None, None, None
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']])
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)
    return X, y, scaler, scaled

X, y, scaler, scaled_data = prepare_data(df)
if X is None:
    st.stop()

# -------------------------------
# 4. Train LSTM model
with st.spinner("🔁 Training LSTM model... Please wait"):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# -------------------------------
# 5. Predict next 7 days
def predict_future(model, scaled_data, scaler, sequence_length=60, days=7):
    input_seq = scaled_data[-sequence_length:]
    forecast = []
    for _ in range(days):
        inp = input_seq.reshape(1, sequence_length, 1)
        pred = model.predict(inp, verbose=0)[0][0]
        forecast.append(pred)
        input_seq = np.append(input_seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

forecast = predict_future(model, scaled_data, scaler)
current_price = df['close'].iloc[-1]

# -------------------------------
# 6. Forecast Graph: Current + Next 7 Days
st.subheader("🔮 Price Forecast: Today + Next 7 Days")

with st.container():
    fig, ax = plt.subplots(figsize=(12, 5))

    # X-axis dates: today + next 7 days
    forecast_dates = pd.date_range(start=df['timestamp'].iloc[-1], periods=8)
    forecast_values = np.insert(forecast, 0, current_price)

    ax.plot(forecast_dates, forecast_values,
            label='🔮 Price Forecast', color='darkorange', marker='o', linestyle='-', linewidth=2.5)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    fig.autofmt_xdate(rotation=45)

    # Labels and title
    ax.set_xlabel("📅 Date", fontsize=14)
    ax.set_ylabel(f"💰 Price ({vs_currency.upper()})", fontsize=14)
    ax.set_title(f"📊 {crypto_id.capitalize()} Forecast (Today + Next 7 Days)", fontsize=16, fontweight='bold')

    # Remove grid and spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(fontsize=12, loc='upper left')

    st.pyplot(fig, use_container_width=True)

# -------------------------------
# 7. Show predicted values
st.subheader("📈 Forecasted Prices:")
st.write(f"Today: **${current_price:.2f}**")
for i, price in enumerate(forecast, 1):
    st.write(f"Day {i}: **${price:.2f}**")
