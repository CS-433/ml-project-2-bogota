import torch
import pickle
import argparse
import numpy as np
from process_data import load_processed_data
from models import NN, CNN, LSTM, Ensemble, RandomForest


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(model_type, dataset):
    # Load train/validation data and move to GPU if available:
    train_lags, train_target, _ = load_processed_data(dataset, partition='train')
    val_lags, val_target, _ = load_processed_data(dataset, partition='val')
    train_lags, train_target = train_lags.to(DEVICE), train_target.to(DEVICE)
    val_lags, val_target = val_lags.to(DEVICE), val_target.to(DEVICE)
    nb_lags = train_lags.shape[1]
    
    # Define number of periods per year to compute performance metrics:
    if dataset in ['BTC-USD', 'ETH-USD', 'XRP-USD']:
        n_periods = 365
    else:
        n_periods = 252
    
    # Find best learning rate for torch models:
    if model_type in ['NN', 'CNN', 'LSTM']:
        best_lr = find_best_lr(model_type, nb_lags, train_lags, train_target, val_lags, val_target, n_periods)
        print()
        print('The best performances of {} on {} are achieved when lr = {:.4f}.'.format(model_type, dataset, best_lr))
        save_params(model_type, dataset, nb_lags, best_lr)
    
    # Find best number of max features for random forest model:
    elif model_type == 'RandomForest':
        best_max_features = find_best_max_features(train_lags, train_target, val_lags, val_target, n_periods)
        print()
        print('The best performances of {} on {} are achieved when max_features = {:.2f}.'.format(model_type, dataset, best_max_features))
        save_params(model_type, dataset, nb_lags, best_max_features)

    else:
        raise ValueError("Unexpected 'model_type' argument: 'model_type' should be in ['NN', 'CNN', 'LSTM', 'RandomForest'].")


def find_best_lr(model_type, nb_lags, train_lags, train_target, val_lags, val_target, n_periods):
    best_hit_rate = 0
    
    # Search the best learning rate on validation set:
    for lr in np.concatenate((np.linspace(start=1e-4, stop=1e-3, num=9, endpoint=False), np.linspace(start=1e-3, stop=1e-2, num=10))):
        # Instantiate model:
        if model_type == 'NN':
            model = Ensemble(model_type=NN, nb_models=10, nb_lags=nb_lags, lr=lr)
        if model_type == 'CNN':
            model = Ensemble(model_type=CNN, nb_models=10, nb_lags=nb_lags, lr=lr)
        if model_type == 'LSTM':
            model = Ensemble(model_type=LSTM, nb_models=10, nb_lags=nb_lags, lr=lr)
        
        # Train model and test it on validation set:
        model.train(train_lags, train_target, num_epochs=5, val_lags=val_lags, val_target=val_target)
        model.test(val_lags, val_target, n_periods=n_periods)
        
        if model.hit_rate > best_hit_rate:
            best_hit_rate = model.hit_rate
            best_lr = lr
    
    return best_lr


def find_best_max_features(train_lags, train_target, val_lags, val_target, n_periods):
    best_hit_rate = 0

    # Convert train/validation data to numpy:
    train_lags, train_target = train_lags.to('cpu').numpy(), train_target.to('cpu').numpy()
    val_lags, val_target = val_lags.to('cpu').numpy(), val_target.to('cpu').numpy()

    # Search the best maximun number of features on validation set:
    for max_features in np.linspace(start=0.1, stop=1.0, num=10):
        # Instantiate model:
        model = RandomForest(n_estimators=1000, random_state=0, max_features=max_features)
        
        # Train model and test it on validation set:
        model.train(train_lags, train_target)
        model.test(val_lags, val_target, n_periods=n_periods)
        
        if model.hit_rate > best_hit_rate:
            best_hit_rate = model.hit_rate
            best_max_features = max_features
    
    return best_max_features


def save_params(model_type, dataset, nb_lags, param):
    if dataset in ['BTC-USD', 'ETH-USD', 'XRP-USD']:
        params_path = '../bestmodels/' + 'cryptos/' + '{}/'.format(dataset) + 'params_{}.pkl'.format(model_type)
    
    elif dataset in ['LBMA-GOLD', 'NYMEX-NG', 'OPEC-ORB']:
        params_path = '../bestmodels/' + 'commodities/' + '{}/'.format(dataset) + 'params_{}.pkl'.format(model_type)
    
    elif dataset in ['SP500', 'CAC40', 'SMI']:
        params_path = '../bestmodels/' + 'stock_market_index/' + '{}/'.format(dataset) + 'params_{}.pkl'.format(model_type)
    
    with open(params_path, "wb") as f:
        pickle.dump((nb_lags, param), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the best parameters of a given model trained on a given dataset.')
    parser.add_argument('model_type', metavar='model_type', type=str, help='An string defining the model type to use.')
    parser.add_argument('dataset', metavar='dataset', type=str, help='A string defining the name of the dataset to use.')
    args = parser.parse_args()
    
    main(args.model_type, args.dataset)