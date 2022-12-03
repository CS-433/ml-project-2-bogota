import numpy as np
import pandas as pd


def get_log_returns(model, input, target, rf):
    output = model(input)
    buy_signal = output > np.log(1+rf)/365
    sell_signal = output < np.log(1+rf)/365
    
    returns = np.zeros_like(target)
    returns[buy_signal] = target[buy_signal]
    returns[sell_signal] = -target[sell_signal]
    return returns


def get_returns(model, input, target, rf):
    output = model(input)
    buy_signal = output > np.log(1+rf)/365
    sell_signal = output < np.log(1+rf)/365
    
    returns = np.zeros_like(target)
    returns[buy_signal] = np.exp(target[buy_signal])-1
    returns[sell_signal] = -(np.exp(target[sell_signal])-1)
    return returns


def compute_hit_rate(model, input, target, rf):
    output = model(input)
    buy_signal = output > np.log(1+rf)/365
    sell_signal = output < np.log(1+rf)/365
    true_buy_signal = target > np.log(1+rf)/365
    true_sell_signal = target < np.log(1+rf)/365
    
    #hit_rate = output[buy_signal].sign().eq(target[buy_signal].sign()).sum() / len(output[buy_signal])
    # hit_rate = output.sign().eq(target.sign()).sum() / len(target)
    # return hit_rate.item()
    nb_buy_hit  = (buy_signal & true_buy_signal).sum()
    nb_sell_hit  = (sell_signal & true_sell_signal).sum()
    nb_position = buy_signal.sum() + sell_signal.sum()
    hit_rate = (nb_buy_hit+nb_sell_hit) / nb_position
    return hit_rate.item()
    #return nb_buy_hit/buy_signal.sum()


def compute_max_drawdown(returns):
    returns = pd.Series(returns)
    cumulative = (returns + 1).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    dd = (cumulative / peak) - 1
    return dd.min()


def compute_sharpe(log_returns, rf, n):
    log_returns = pd.Series(log_returns)
    mean = log_returns.mean()
    std = log_returns.std()
    return (n * mean - np.log(1+rf))/(std*np.sqrt(n))