import os
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from datetime import datetime

# 1. Load credentials
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# 2. Initialize the Client
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def fetch_stock_data(symbol, start_date, end_date):
    print(f"--- Fetching data for {symbol} ---")

    # Define the request parameters
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Hour,  # Hourly data is great for day-trading models
        start=datetime.strptime(start_date, '%Y-%m-%d'),
        end=datetime.strptime(end_date, '%Y-%m-%d')
    )

    # Fetch the bars
    bars = client.get_stock_bars(request_params)

    # Convert to a Pandas DataFrame
    df = bars.df

    # Reset index to move (symbol, timestamp) into columns
    df = df.reset_index()

    # Keep only the columns we need for ML
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Save to our data folder
    output_path = f"data/{symbol}_historical.csv"
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Example: Fetching 2 years of Apple data
    fetch_stock_data("AAPL", "2025-06-01", "2026-05-01")
