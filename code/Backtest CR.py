# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:43:50 2023

@author: Sea
"""

import os

file_list = []
path_of_the_directory = r"C:\Users\85296\Downloads\Quant-Parity3"
for filename in os.listdir(path_of_the_directory):
    f = os.path.join(path_of_the_directory, filename)
    if os.path.isfile(f):
        file_list.append(f)

result_list = []

import pandas as pd
import numpy as np
import scipy

for i in file_list:

    try:
        xgboost_result = i
        xgboost_data = i[39:52]
        fx_pair = i[43:49]
        month_period = int(i[50:51])
        fx_link = r"C:\Users\85296\Downloads\FX CR-20230331T134515Z-001\FX CR\PPPBBBCR.csv"
        fx_raw = fx_link.replace("PPPBBB", fx_pair)

        # xgboost data
        xgboost = pd.read_csv(xgboost_result)
        xgboost["end_date"] = pd.to_datetime(xgboost["end_date"])
        # as a signal at t=0 means we buy the fx at t=0 and sell at t=1, we shift the column so that we can backtest by pct_change
        xgboost.iloc[:, 1] = xgboost.iloc[:, 1].shift(1)
        # now, signal at t=0 means we open the position at t-1 to t=0
        xgboost.set_index("end_date", inplace=True)
        # as only can 1, 0 for binary classification, we chnage 0 to -1 for sell signal
        xgboost["0"] = xgboost["0"].replace(0, -1)
        if len(xgboost) < 10:
            continue

        # fx data
        fx = pd.read_csv(fx_raw)
        fx = fx.drop(fx.columns[[1, 2]], axis=1)
        fx.drop(index=fx.index[0], axis=0, inplace=True)
        fx["Unnamed: 0"] = pd.to_datetime(fx["Unnamed: 0"])
        fx.set_index("Unnamed: 0", inplace=True)
        fx.iloc[:, 0] = [float(x) for x in fx.iloc[:, 0]]
        # fill Na value with back filling, we trade the fx for next business day if the last date of the month is weekend
        fx = fx.asfreq("d")
        fx = fx.fillna(method="bfill")

        # backtesting
        backtest_df = pd.merge(xgboost, fx, left_index=True, right_index=True)
        backtest_df.columns = ["signal", "price"]

        transaction_cost = 0.00  # 0% transaction cost

        # Calculate the percentage change in price
        backtest_df['pct_change'] = backtest_df["price"].pct_change()

        # Calculate the PnL for each trade
        backtest_df['PnL'] = np.nan
        buy_mask = backtest_df['signal'] == 1
        sell_mask = backtest_df['signal'] == -1
        hold_mask = backtest_df['signal'] == 0
        backtest_df.loc[buy_mask, 'PnL'] = 1 + backtest_df.loc[buy_mask, 'pct_change'] - transaction_cost
        backtest_df.loc[sell_mask, 'PnL'] = 1 - backtest_df.loc[sell_mask, 'pct_change'] - transaction_cost
        backtest_df.loc[hold_mask, 'PnL'] = 1

        # Calculate the cumulative return
        backtest_df['cumulative_return'] = (backtest_df['PnL']).cumprod()

        # Calculate the annualized return
        start_date = pd.to_datetime(backtest_df.index.values[0])
        end_date = pd.to_datetime(backtest_df.index.values[-1])
        n_days = (end_date - start_date).days
        n_years = n_days / 365
        total_return = backtest_df['cumulative_return'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        # Calculate the period return annualized volatility
        period_return = backtest_df['PnL'].dropna().tolist()
        annualized_volatility = np.std(period_return) * np.sqrt(12 / month_period)

        # Calculate the period return skewness
        period_return_skewness = scipy.stats.skew(period_return, bias=False)

        # Calculate the Sharpe Ratio
        risk_free_rate = 0.01
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

        # Calculate the max drawdown
        cumulative_returns = backtest_df['cumulative_return']
        previous_peaks = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()

        # Calculate the Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown)

        print(xgboost_data)
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Period Return Annualized Volatility: {annualized_volatility:.2%}")
        print(f"Period Return Skewness: {period_return_skewness:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")

        # backtest_df.to_csv("result.csv")

        import matplotlib.pyplot as plt

        plt.plot(backtest_df.index.values, backtest_df['cumulative_return'])
        plt.title(xgboost_data + 'cumulative_return')
        plt.show()

        plt.plot(backtest_df.index.values, backtest_df['PnL'])
        plt.title(xgboost_data + 'PnL')
        plt.show()

        result_list.append(
            [xgboost_data, annualized_return, annualized_volatility, period_return_skewness, sharpe_ratio, max_drawdown,
             calmar_ratio])
    except:
        print("error")

result_df = pd.DataFrame(result_list)
result_df.columns = ['xgboost_data', 'Annualized Return', 'Period Return Annualized Volatility',
                     'Period Return Skewness', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']
result_df.set_index('xgboost_data', inplace=True)

result_df_data = result_df.describe()

result_df.to_csv("result_df.csv")

result_df_data.to_csv("result_df_data.csv")

import seaborn as sns

heatmap_df = pd.DataFrame()

for i in range(len(result_df)):
    month = result_df.index[i].split("_")[2]
    pairs = result_df.index[i].split("_")[0] + result_df.index[i].split("_")[1]
    heatmap_df.loc[month, pairs] = result_df.iloc[i, 0]

heatmap_df = heatmap_df.replace(np.nan, 0)

sns.heatmap(heatmap_df)