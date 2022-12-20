import numpy as np
import pandas as pd
import torch


RF = 0


def get_log_returns(model, lags, target, rf=RF):
    output = model.forecast(lags)
    output = torch.Tensor(output)
    return (output.sign() * target).detach().to('cpu').numpy()


def get_returns(model, lags, target, rf=RF):
    log_returns = get_log_returns(model, lags, target)
    return np.exp(log_returns) - 1


def compute_hit_rate(model, lags, target, rf=RF):
    output = model.forecast(lags)
    output, target = torch.Tensor(output), torch.Tensor(target)
    hit_position = (output.sign() * target.sign()) >= 0
    return (hit_position.sum() / len(hit_position)).item()


def compute_max_drawdown(returns):
    returns = pd.Series(returns)
    cumulative = (returns + 1).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    dd = (cumulative / peak) - 1
    return dd.abs().max()


def compute_sharpe(log_returns, n_periods, rf=RF):
    log_returns = pd.Series(log_returns)
    mean = log_returns.mean()
    std = log_returns.std()
    return (n_periods * mean - np.log(1 + rf)) / (std * np.sqrt(n_periods))