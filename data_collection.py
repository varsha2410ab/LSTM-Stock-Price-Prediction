import yfinance as yf
import pandas as pd

def fetch_data(ticker="AAPL", start="2022-01-01", end="2023-01-01", file_path="data.csv"):
    data = yf.download(ticker, start=start, end=end)

    # Fix: Flatten MultiIndex columns (if any)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Reset index to make 'Date' a column
    data.reset_index(inplace=True)

    # Now save to CSV
    data.to_csv(file_path, index=False, header=True)
