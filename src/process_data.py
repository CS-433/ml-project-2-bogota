import torch
import quandl
import pandas as pd
import numpy as np

DATA_FOLDER = '../data/'


def load_data(path):
    # df = pd.read_csv(path)
    # df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    # df['Date'] = pd.to_datetime(df['Date'])

    # df['log_return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    # df.drop(columns='Adj Close', inplace=True)
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.dropna(inplace=True)
    df = quandl.get("BCHARTS/BITSTAMPUSD", start_date="2014-04-15", end_date="2019-01-10")
    print(df.columns)
    df['log_return'] = np.log(df['Open']/df['Open'].shift(1))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def split_data(df, train_ratio):
    date = df.index[int(train_ratio*len(df))]
    train_period = (df.index < date)
    val_period = (df.index >= date)

    df_train = df.loc[train_period]
    df_val = df.loc[val_period]
    return df_train, df_val


def create_sequences(df, nb_lags):
    sequences = torch.empty((len(df)-nb_lags, nb_lags))
    targets = torch.empty(len(df)-nb_lags)
    
    for i in range(len(df)-nb_lags):
        sequences[i,:] = torch.tensor(df['log_return'].iloc[i:i+nb_lags].values)
        targets[i] = torch.tensor(df['log_return'].iloc[i+nb_lags])
    
    return sequences, targets


if __name__ == '__main__':

    df = load_data(DATA_FOLDER + 'BTC-USD.csv')
    df_train, df_val = split_data(df, train_ratio=0.4482)
    train_input, train_target = create_sequences(df_train, nb_lags=6)
    val_input, val_target = create_sequences(df_val, nb_lags=6)
    
    with open(DATA_FOLDER + 'train_data.pkl', "wb") as f:
        torch.save((train_input, train_target), f)
    
    with open(DATA_FOLDER + 'val_data.pkl', "wb") as f:
        torch.save((val_input, val_target), f)