import re
import torch
import pandas as pd
import numpy as np
import argparse


DATA_FOLDER = "../data/"


def main(dataset, nb_lags, train_ratio):
    read_path, save_path = find_paths(dataset)
    df = load_data(read_path, dataset)
    df_train, df_val = split_data(df, train_ratio=train_ratio)
    train_input, train_target = create_sequences(df_train, nb_lags=nb_lags)
    val_input, val_target = create_sequences(df_val, nb_lags=nb_lags)

    with open(save_path + "train_data.pkl", "wb") as f:
        torch.save((train_input, train_target), f)

    with open(save_path + "val_data.pkl", "wb") as f:
        torch.save((val_input, val_target), f)


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
        raise ValueError("Unexpected 'dataset' argument when running process_data.py: 'dataset' should be in ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LBMA-GOLD', 'NYMEX-NG', 'OPEC-ORB', 'SP500', 'CAC40', 'SMI'].")
    
    return read_path, save_path


def load_data(path, dataset):
    if dataset in ['BTC-USD', 'ETH-USD', 'XRP-USD']:
        df = pd.read_csv(path, header=1)
        df['Date'] = pd.to_datetime(df.date)
        df['Price'] = df['open']
    
    elif dataset == 'LBMA-GOLD':
        df = pd.read_csv(path)
        df.drop(columns=['USD (PM)', 'GBP (AM)', 'GBP (PM)', 'EURO (AM)', 'EURO (PM)'], inplace=True)
        df.rename(columns={'USD (AM)':'Price'}, inplace=True)
        df['Date'] = pd.to_datetime(df.Date)
    
    elif dataset == 'NYMEX-NG':
        df = pd.read_csv(path)
        df['Price'] = df['Open']
        df['Date'] = pd.to_datetime(df.Date)
    
    elif dataset == 'OPEC-ORB':
        df = pd.read_csv(path)
        df.rename(columns={'Value':'Price'}, inplace=True)
        df['Date'] = pd.to_datetime(df.Date)
    
    elif dataset == 'SP500':
        df = pd.read_csv(path)
        df['Open'] = df['Open'].apply(lambda x: re.sub(",", "", x))
        df['Price'] = pd.to_numeric(df['Open'])
        df['Date'] = pd.to_datetime(df.Date)

    elif dataset in ['CAC40', 'SMI']:
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df.Date)
        df['Price'] = df['Open']
    
    df = df[['Date', 'Price']]
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df["Log_Return"] = np.log(df["Price"] / df["Price"].shift(1))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def split_data(df, train_ratio):
    date = df.index[int(train_ratio * len(df))]
    train_period = df.index < date
    val_period = df.index >= date

    df_train = df.loc[train_period]
    df_val = df.loc[val_period]
    return df_train, df_val


def create_sequences(df, nb_lags):
    sequences = torch.empty((len(df) - nb_lags, nb_lags))
    targets = torch.empty(len(df) - nb_lags)

    for i in range(len(df) - nb_lags):
        sequences[i, :] = torch.tensor(df["Log_Return"].iloc[i : i + nb_lags].values)
        targets[i] = torch.tensor(df["Log_Return"].iloc[i + nb_lags])

    return sequences, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data for the given dataset, i.e., load the data, compute the Log-Returns, split the data in train and test set, create sequences with the given number of lags and finally save the sequences.')
    parser.add_argument('dataset', metavar='dataset', type=str, help='A string defining the name of the dataset to process.')
    parser.add_argument('nb_lags', metavar='nb_lags', type=int, help='An int defining the number of lags.')
    parser.add_argument('train_ratio', metavar='train_ratio', type=float, help='A float defining the train ratio.')
    args = parser.parse_args()
    
    main(args.dataset, args.nb_lags, args.train_ratio)
