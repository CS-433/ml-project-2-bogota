import argparse
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from process_data import load_processed_data
from models import NN, CNN, LSTM, Ensemble, RandomForest


sns.set()
sns.set_style("whitegrid")
colors = sns.color_palette("colorblind", 16)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(model_type, dataset):
    # Load train/validation data and move to GPU if available:
    train_lags, train_target, _ = load_processed_data(dataset, partition='train')
    val_lags, val_target, _ = load_processed_data(dataset, partition='val')
    train_lags, train_target = train_lags.to(DEVICE), train_target.to(DEVICE)
    val_lags, val_target = val_lags.to(DEVICE), val_target.to(DEVICE)
    
    # Load best param (lr or max_features):
    nb_lags, param = load_best_params(model_type, dataset)
    
    # Instantiate model and train:
    if model_type == 'NN':
        model = Ensemble(model_type=NN, nb_models=100, nb_lags=nb_lags, lr=param)
        model.train(train_lags, train_target, num_epochs=500, val_lags=val_lags, val_target=val_target)
        
    elif model_type == 'CNN':
        model = Ensemble(model_type=CNN, nb_models=100, nb_lags=nb_lags, lr=param)
        model.train(train_lags, train_target, num_epochs=500, val_lags=val_lags, val_target=val_target)
        
    elif model_type == 'LSTM':
        model = Ensemble(model_type=LSTM, nb_models=100, nb_lags=nb_lags, lr=param)
        model.train(train_lags, train_target, num_epochs=500, val_lags=val_lags, val_target=val_target)
    
    elif model_type == 'RandomForest':
        pass
    
    else:
        raise ValueError("Unexpected 'model_type' argument: 'model_type' should be in ['NN', 'CNN', 'LSTM', 'RandomForest'].")
    
    # Save model:
    if model_type in ['NN', 'CNN', 'LSTM']:
        save_model(model, model_type, dataset)
    
    # Plot losses and hit_rates for torch models:
    if model_type in ['NN', 'CNN', 'LSTM']:
        plot_metrics(model, model_type, dataset)


def load_best_params(model_type, dataset):
    if dataset in ['BTC-USD', 'ETH-USD', 'XRP-USD']:
        params_path = '../bestmodels/' + 'cryptos/' + '{}/'.format(dataset) + 'params_{}.pkl'.format(model_type)
    
    elif dataset in ['LBMA-GOLD', 'NYMEX-NG', 'OPEC-ORB']:
        params_path = '../bestmodels/' + 'commodities/' + '{}/'.format(dataset) + 'params_{}.pkl'.format(model_type)
    
    elif dataset in ['SP500', 'CAC40', 'SMI']:
        params_path = '../bestmodels/' + 'stock_market_index/' + '{}/'.format(dataset) + 'params_{}.pkl'.format(model_type)
    
    with open(params_path, "rb") as f:
        nb_lags, param = pickle.load(f)
    
    return nb_lags, param


def save_model(model, model_type, dataset):
    if model_type not in ['NN', 'CNN', 'LSTM', 'RandomForest']:
        raise ValueError("Unexpected 'model_type' argument: 'model_type' should be in ['NN', 'CNN', 'LSTM', 'RandomForest'].")
    
    if dataset in ['BTC-USD', 'ETH-USD', 'XRP-USD']:
        model_path = '../bestmodels/' + 'cryptos/' + '{}/'.format(dataset) + '{}.pkl'.format(model_type)
    
    elif dataset in ['LBMA-GOLD', 'NYMEX-NG', 'OPEC-ORB']:
        model_path = '../bestmodels/' + 'commodities/' + '{}/'.format(dataset) + '{}.pkl'.format(model_type)
    
    elif dataset in ['SP500', 'CAC40', 'SMI']:
        model_path = '../bestmodels/' + 'stock_market_index/' + '{}/'.format(dataset) + '{}.pkl'.format(model_type)
    
    else:
        raise ValueError("Unexpected 'dataset' argument: 'dataset' should be in ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LBMA-GOLD', 'NYMEX-NG', 'OPEC-ORB', 'SP500', 'CAC40', 'SMI'].")
    
    with open(model_path, "wb") as f:
        model.move_to('cpu')
        torch.save(model, f)


def plot_metrics(model, model_type, dataset):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Losses and hit-rates during training of {} on {}'.format(model_type, dataset), fontsize=14)
    plt.subplots_adjust(wspace= 0.25, hspace= 0.25)
    
    for model in model.models:
        ax1.plot(model.train_losses, color=colors[0])
        ax1.plot(model.val_losses, color=colors[1])
        
        ax2.plot(model.train_hit_rates, color=colors[0])
        ax2.plot(model.val_hit_rates, color=colors[1])
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Losses')
    ax1.legend(['Train set', 'Validation set'])
    ax1.grid(True)
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Hit-Rates')
    ax2.legend(['Train set', 'Validation set'])
    ax2.grid(True)
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model of a given model type on a given dataset with the best parameters saved by validation.py')
    parser.add_argument('model_type', metavar='model_type', type=str, help='An string defining the model type to use.')
    parser.add_argument('dataset', metavar='dataset', type=str, help='A string defining the name of the dataset to use.')
    args = parser.parse_args()
    
    main(args.model_type, args.dataset)