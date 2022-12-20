import pickle
import torch
import pandas as pd
import numpy as np
import argparse


DATA_FOLDER = "../data/"


def main(dataset, nb_lags, train_ratio):
    read_path, save_path = find_paths(dataset)
    df = load_data(read_path, dataset)
    df_train, df_val, df_test = split_data(df, train_ratio=train_ratio)
    train_lags, train_target, train_dates = create_sequences(df_train, nb_lags=nb_lags)
    val_lags, val_target, val_dates = create_sequences(df_val, nb_lags=nb_lags)
    test_lags, test_target, test_dates = create_sequences(df_test, nb_lags=nb_lags)
    
    with open(save_path + "train_data.pkl", "wb") as f:
        pickle.dump((train_lags, train_target, train_dates), f)

    with open(save_path + "val_data.pkl", "wb") as f:
        pickle.dump((val_lags, val_target, val_dates), f)
    
    with open(save_path + "test_data.pkl", "wb") as f:
        pickle.dump((test_lags, test_target, test_dates), f)


def find_paths(dataset):
    if dataset in ['BTC-USD', 'ETH-USD', 'XRP-USD']:
        read_path = DATA_FOLDER + 'raw/' + 'cryptos/' + '{}.csv'.format(dataset)
        save_path = DATA_FOLDER + 'processed/' + 'cryptos/' + '{}/'.format(dataset)
    
    elif dataset in ['LBMA-GOLD', 'NYMEX-NG', 'OPEC-ORB']:
        read_path = DATA_FOLDER + 'raw/' + 'commodities/' + '{}.csv'.format(dataset)
        save_path = DATA_FOLDER + 'processed/' + 'commodities/' + '{}/'.format(dataset)
    
    elif dataset in ['SP500', 'CAC40', 'SMI']:
        read_path = DATA_FOLDER + 'raw/' + 'stock_market_index/' + '{}.csv'.format(dataset)
        save_path = DATA_FOLDER + 'processed/' + 'stock_market_index/' + '{}/'.format(dataset)
    
    else:
        raise ValueError("Unexpected 'dataset' argument: 'dataset' should be in ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LBMA-GOLD', 'NYMEX-NG', 'OPEC-ORB', 'SP500', 'CAC40', 'SMI'].")
    
    return read_path, save_path


def load_data(path, dataset):
    df = pd.read_csv(path)
    
    # Data from: https://www.cryptodatadownload.com/data/bitstamp/
    if dataset in ['BTC-USD', 'ETH-USD', 'XRP-USD']:
        df['Date'] = pd.to_datetime(df.date)
        df['Price'] = df['open']
    
    # Data from: https://data.nasdaq.com/data/LBMA/GOLD-gold-price-london-fixing
    elif dataset == 'LBMA-GOLD':
        df.drop(columns=['USD (PM)', 'GBP (AM)', 'GBP (PM)', 'EURO (AM)', 'EURO (PM)'], inplace=True)
        df.rename(columns={'USD (AM)':'Price'}, inplace=True)
        df['Date'] = pd.to_datetime(df.Date)
    
    # Data from: https://www.nasdaq.com/market-activity
    elif dataset in ['NYMEX-NG', 'SP500']:
        df['Price'] = df['Open']
        df['Date'] = pd.to_datetime(df.Date)
    
    # Data from: https://data.nasdaq.com/data/OPEC/ORB-opec-crude-oil-price
    elif dataset == 'OPEC-ORB':
        df.rename(columns={'Value':'Price'}, inplace=True)
        df['Date'] = pd.to_datetime(df.Date)

    # Data from: https://finance.yahoo.com/
    elif dataset in ['CAC40', 'SMI']:
        df['Date'] = pd.to_datetime(df.Date)
        df['Price'] = df['Open']
    
    df = df[['Date', 'Price']]
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    np.seterr(divide = 'ignore')
    df["Log_Return"] = np.log(df["Price"] / df["Price"].shift(1))
    np.seterr(divide = 'warn') 
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def split_data(df, train_ratio):
    date = df.index[int(train_ratio * len(df))]
    train_period = df.index < date
    val_period = df.index >= date

    df_train = df.loc[train_period]
    df_val = df.loc[val_period]
    df_test = df_val.iloc[int(len(df_val)/2):]
    df_val = df_val.iloc[:int(len(df_val)/2)]
    
    return df_train, df_val, df_test


def create_sequences(df, nb_lags):
    sequences = torch.empty((len(df) - nb_lags, nb_lags))
    targets = torch.empty(len(df) - nb_lags)
    dates = []
    
    for i in range(len(df) - nb_lags):
        sequences[i, :] = torch.tensor(df["Log_Return"].iloc[i : i + nb_lags].values)
        targets[i] = torch.tensor(df["Log_Return"].iloc[i + nb_lags])
        dates.append(df.index[i + nb_lags])
    
    dates = pd.Series(dates)
    
    return sequences, targets, dates


def load_processed_data(dataset, partition):
    _, save_path = find_paths(dataset)
    with open(save_path + "{}_data.pkl".format(partition), "rb") as f:
        lags, target, dates = pickle.load(f)
    return lags, target, dates


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process data for the given dataset, i.e., load the data, compute the Log-Returns, split the data in train and test set, create sequences with the given number of lags and finally save the sequences.')
    parser.add_argument('dataset', metavar='dataset', type=str, help='A string defining the name of the dataset to process.')
    parser.add_argument('nb_lags', metavar='nb_lags', type=int, help='An int defining the number of lags.')
    parser.add_argument('train_ratio', metavar='train_ratio', type=float, help='A float defining the train ratio.')
    args = parser.parse_args()
    
    main(args.dataset, args.nb_lags, args.train_ratio)