import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --------------------------
# Define tickers
nifty_ticker = "^NSEI"
sensex_ticker = "^BSESN"

# --------------------------
# Technical indicators
def calculate_technical_indicators(data):
    """Calculate SMA, EMA, RSI, MACD"""
    data["SMA"] = data["Close"].rolling(window=10).mean()
    data["EMA"] = data["Close"].ewm(span=10, adjust=False).mean()
    
    # RSI calculation (7-day)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(7).mean()
    avg_loss = loss.rolling(7).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    
    return data

# --------------------------
# Prepare ML data
def prepare_data_for_ml(data):
    indicators = ["SMA", "EMA", "RSI", "MACD"]
    
    # Drop rows with NaN in any indicator
    data_clean = data[indicators + ["Close"]].dropna()
    
    X = data_clean[indicators]
    y = (data_clean["Close"].shift(-1) > data_clean["Close"]).astype(int)
    
    # Align X and y
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    print("DEBUG: X rows:", len(X), "Y rows:", len(y))  # check rows
    return X, y

# --------------------------
# Train ML model
def train_model(X, y):
    if len(X) == 0:
        raise ValueError("No training data available. Check indicator calculation and dropna().")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# --------------------------
# Suggest trades
def suggest_trades(model, data):
    indicators = ["SMA", "EMA", "RSI", "MACD"]
    X = data[indicators].dropna()
    predictions = model.predict(X)
    trades = []
    last_signal = None
    for i, pred in enumerate(predictions):
        signal = "Buy" if pred == 1 else "Sell"
        if signal != last_signal:
            trades.append((data.index[i], signal))
        last_signal = signal
    return trades

# --------------------------
# Main
# Download Close prices (Adj Close issue fixed)
nifty_data = yf.download(nifty_ticker, period="5y", progress=False)[["Close"]]
sensex_data = yf.download(sensex_ticker, period="5y", progress=False)[["Close"]]

# Calculate indicators
nifty_data = calculate_technical_indicators(nifty_data)
sensex_data = calculate_technical_indicators(sensex_data)

# Prepare ML data
nifty_X, nifty_y = prepare_data_for_ml(nifty_data)
sensex_X, sensex_y = prepare_data_for_ml(sensex_data)

# Train models
nifty_model, nifty_accuracy = train_model(nifty_X, nifty_y)
sensex_model, sensex_accuracy = train_model(sensex_X, sensex_y)

print(f"NIFTY Model Accuracy: {nifty_accuracy:.2f}")
print(f"SENSEX Model Accuracy: {sensex_accuracy:.2f}")

# Generate trades
nifty_trades = suggest_trades(nifty_model, nifty_data)
sensex_trades = suggest_trades(sensex_model, sensex_data)

print("\nNIFTY Trades:")
for trade in nifty_trades:
    print(trade)

print("\nSENSEX Trades:")
for trade in sensex_trades:
    print(trade)

# --------------------------
# Plot NIFTY
plt.figure(figsize=(12,6))
plt.plot(nifty_data["Close"], label="NIFTY Close")
plt.plot(nifty_data["SMA"], label="NIFTY SMA")
plt.plot(nifty_data["EMA"], label="NIFTY EMA")
plt.title("NIFTY Technical Indicators")
plt.legend()
plt.show()

# Plot SENSEX
plt.figure(figsize=(12,6))
plt.plot(sensex_data["Close"], label="SENSEX Close")
plt.plot(sensex_data["SMA"], label="SENSEX SMA")
plt.plot(sensex_data["EMA"], label="SENSEX EMA")
plt.title("SENSEX Technical Indicators")
plt.legend()
plt.show()
