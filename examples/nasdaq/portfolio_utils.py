from utils_dir import get_curr_dir, include_home_dir
include_home_dir()

import pandas as pd
import numpy as np

from jumpmodels.utils import filter_date_range        # useful helpers
from jumpmodels.jump import JumpModel                 # class of JM & CJM
from jumpmodels.sparse_jump import SparseJumpModel    # class of Sparse JM
from jumpmodels.plot import plot_regimes_and_cumret, savefig_plt
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd

from get_data import get_data

from feature import DataLoader

from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def calculate_cumulative_return_with_dates(labels, returns, dates):
    cumulative_return = 1.0
    in_market_dates = []
    sell_dates = []
    current_state = None
    first_trade_date = None
    last_trade_date = None
    trading_days = 0

    for i, (label, ret, date) in enumerate(zip(labels, returns, dates)):
        if label == 0:  # Stay in the market
            if current_state != 0:
                in_market_dates.append(date)
                current_state = 0
                if first_trade_date is None:
                    first_trade_date = date
                # Skip the return on the day of buying
                continue
            cumulative_return *= (1 + ret)
            last_trade_date = date
            trading_days += 1
        else:  # Exit the market
            if current_state == 0:
                sell_dates.append(date)
                current_state = 1
                # Include the return on the day of selling
                cumulative_return *= (1 + ret)
                trading_days += 1

    total_return = cumulative_return - 1

    # Annualize the return
    if first_trade_date and last_trade_date:
        years = ((last_trade_date - first_trade_date).days+1) / 365
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = total_return
    else:
        annualized_return = 0

    return total_return, annualized_return, in_market_dates, sell_dates, trading_days


def calc_returns(data, X_processed, labels, last_N_days=False, N=60):
    # Calculate returns
    aligned_returns = data.ret_ser.loc[X_processed.index]
    dates = X_processed.index

    if last_N_days:
        # Get the last N days of data
        last_N_days_data = aligned_returns.iloc[-N:]
        last_N_days_labels = labels[-N:]
        last_N_days_dates = dates[-N:]
        
        total_return, annualized_return, in_market_dates, sell_dates, trading_days = calculate_cumulative_return_with_dates(
            last_N_days_labels, last_N_days_data, last_N_days_dates)
        
        buy_and_hold_return = (1 + last_N_days_data[-(N+1):]).prod() - 1
        years = N / 365  # Assuming 60 trading days is approximately 60 calendar days
    else:
        total_return, annualized_return, in_market_dates, sell_dates, trading_days = calculate_cumulative_return_with_dates(
            labels, aligned_returns, dates)
        
        buy_and_hold_return = (1 + aligned_returns).prod() - 1
        years = ((aligned_returns.index[-1] - aligned_returns.index[0]).days) / 365

    # Calculate annualized buy-and-hold return
    annualized_buy_and_hold_return = (1 + buy_and_hold_return) ** (1 / years) - 1

    # Print the results
    print(f"{'Last {N} days ' if last_N_days else ''}Total cumulative return: {total_return:.2%}")
    print(f"{'Last {N} days ' if last_N_days else ''}Annualized return: {annualized_return:.2%}")
    print(f"{'Last {N} days ' if last_N_days else ''}Buy-and-hold return: {buy_and_hold_return:.2%}")
    print(f"{'Last {N} days ' if last_N_days else ''}Annualized buy-and-hold return: {annualized_buy_and_hold_return:.2%}")
    print(f"{'Last {N} days ' if last_N_days else ''}Number of trading days: {trading_days}")

    print("\nDates to enter the market:")
    for date in in_market_dates:
        print(date.strftime('%Y-%m-%d'))
    print("\nDates to sell:")
    for date in sell_dates:
        print(date.strftime('%Y-%m-%d'))



def process_updates(ticker):
    # Get today's date
    today = pd.Timestamp.now().date()

    # load data
    get_data(ticker, start="1985-10-01", end=today.strftime('%Y-%m-%d'))
    data = DataLoader(ticker=ticker, ver="v0").load(start_date="2007-1-1", end_date=today.strftime('%Y-%m-%d'))

    train_start, test_start = "2007-01-01", "2024-10-13"
    
    # Check if the data starts later than train_start
    print(data.X.index[0], pd.Timestamp(train_start))
    if data.X.index[0] > pd.Timestamp(train_start).date():
        train_start = (data.X.index[0] + timedelta(days=2)).strftime('%Y-%m-%d')
        print(f"Data starts later than requested. New train_start: {train_start}")

    # filter dates
    X_train = filter_date_range(data.X, start_date=train_start, end_date=test_start)
    X_test = filter_date_range(data.X, start_date=train_start)
    
    # print time split
    train_start, train_end = X_train.index[[0, -1]]
    test_start, test_end = X_test.index[[0, -1]]
    print(f"{ticker} Training starts at:", train_start, "and ends at:", train_end)
    print(f"{ticker} Testing starts at:", test_start, "and ends at:", test_end)

    # Preprocessing
    clipper = DataClipperStd(mul=3.)
    scalar = StandardScalerPD()
    # fit on training data
    X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
    # transform the test data
    X_test_processed = scalar.transform(clipper.transform(X_test))

    # set the jump penalty
    jump_penalty=50.
    # initlalize the JM instance
    jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )

    jm.fit(X_train_processed, data.ret_ser, sort_by="cumret")

    # save training plot
    ax, ax2 = plot_regimes_and_cumret(jm.labels_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
    ax.set(title=f"In-Sample Fitted Regimes by the JM ($\\lambda$={jump_penalty})")
    print(f"{get_curr_dir()}/portfolio/{ticker}/JM_lambd-{jump_penalty}_train.pdf")
    savefig_plt(f"{get_curr_dir()}/portfolio/{ticker}/JM_lambd-{jump_penalty}_train.pdf")

    # refit
    jump_penalty=50.
    jm.set_params(jump_penalty=jump_penalty).fit(X_train_processed, data.ret_ser, sort_by="cumret")
    # make online inference 
    # labels_test_online = jm.predict_online(X_test_processed)

    # make inference using all test data
    labels_test = jm.predict(X_test_processed)
    N=300
    # print(labels_test)
    # plot
    ax, ax2 = plot_regimes_and_cumret(labels_test, data.ret_ser, n_c=2, start_date=test_start, end_date=test_end, last_N_days=True, N=N)
    # Set major locator to show every day
    # ax.xaxis.set_major_locator(mdates.DayLocator())

    # # Format the date to show as 'YYYY-MM-DD'
    # date_formatter = mdates.DateFormatter('%d')
    # ax.xaxis.set_major_formatter(date_formatter)

    # # Rotate and align the tick labels so they look better
    # plt.gcf().autofmt_xdate()
    _ = ax.set(title=f"Out-of-Sample Predicted Regimes by the JM Using All Test Data ($\\lambda$={jump_penalty})")
    savefig_plt(f"{get_curr_dir()}/portfolio/{ticker}/JM_lambd-{jump_penalty}_test_online.pdf")

    # calc returns
    calc_returns(data, X_test_processed, labels_test, last_N_days=True, N=N)
    return
