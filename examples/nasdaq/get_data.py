"""
This script retrieves the daily closing price data for 
the Nasdaq-100 index from Yahoo Finance via its Python API.

Users do not need to run this script manually, as the return data 
is already saved in `example/Nasdaq/data/`. 
"""

from utils_dir import *
include_home_dir()

import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

from jumpmodels.utils import check_dir_exist

def get_data(ticker, start="1985-10-01", end="2025-01-10"):
    # download closing prices
    # close: pd.Series = yf.download("^"+TICKER, start="1985-10-01", end="2024-12-26")['Close']
    close: pd.Series = yf.download(ticker, start=start, end=end)['Close']
    # convert to ret
    ret = close.pct_change()
    # concat as df
    try:
        df = pd.DataFrame({"close": close.values.reshape(-1), "ret": ret.values.reshape(-1)}, index=close.index.date)
    except ValueError as e:
        print(f"Error creating DataFrame: {e}")
        print(f"Close shape: {close.shape}, Return shape: {ret.shape}")

    df.index.name = "date"

    # save
    curr_dir = get_curr_dir()
    data_dir = f"{curr_dir}/data/"; check_dir_exist(data_dir)
    pd.to_pickle(df, f"{data_dir}{ticker}.pkl")
    np.round(df, 6).to_csv(f"{data_dir}{ticker}.csv")
    print("Successfully downloaded data for ticker:", ticker)
    return 

def pickle(ticker):
    # Define the file paths
    curr_dir = get_curr_dir()
    data_dir = Path(f"{curr_dir}/data/")
    csv_file_path = data_dir / f"{ticker}.csv"
    pickle_file_path = data_dir / f"{ticker}.pkl"

    # Read the CSV file
    df = pd.read_csv(csv_file_path, index_col="date", parse_dates=True)

    # Ensure the index is of type date
    df.index = pd.to_datetime(df.index).date

    # Calculate returns if not already present
    if "ret" not in df.columns:
        df["ret"] = df["open"].pct_change()

    # Round the values to 6 decimal places
    df = np.round(df, 6)

    # Save as pickle file
    df.to_pickle(pickle_file_path)

    print(f"Successfully saved {ticker} data as pickle file.")

if __name__ == "__main__":
    TICKER = "PLTR"   # Nasdaq-100 Index
    get_data(TICKER)
    # pickle(TICKER)