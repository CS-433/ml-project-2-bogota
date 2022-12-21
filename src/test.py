import argparse
import pickle
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from train import load_best_params
from process_data import load_processed_data
from models import RandomForest


sns.set()
sns.set_style("whitegrid")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(model_type, dataset):
    train_lags, train_target, train_dates = load_processed_data(dataset, partition='train')
    val_lags, val_target, val_dates = load_processed_data(dataset, partition='val')
    test_lags, test_target, test_dates = load_processed_data(dataset, partition='test')
    
    if model_type in ['NN', 'CNN', 'LSTM']:
        model = load_pretrained_model(model_type, dataset)
    else:
        # Load best para mmax_features:
        _, param = load_best_params(model_type, dataset)
        model = RandomForest(n_estimators=10000, random_state=0, max_features=param)
        model.train(train_lags, train_target)
        
    # Define number of periods per year to compute performance metrics:
    if dataset in ['BTC-USD', 'ETH-USD', 'XRP-USD']:
        n_periods = 365
    else:
        n_periods = 252
    
    # Plot cumulative sum of model's log-returns:
    fig = plt.figure(figsize=(9, 6))
    fig.suptitle('Performances of {} on {}'.format(model_type, dataset), fontsize=14)
    plt.subplots_adjust(wspace= 0.25, hspace= 0.25)

    sub1 = fig.add_subplot(2,2,(1,2))
    model.test(train_lags, train_target, n_periods)
    sub1.plot(train_dates, np.cumsum(model.log_returns))
    sub1.plot(train_dates, np.cumsum(train_target.to('cpu')))
    sub1.grid(True)
    sub1.set_ylabel('Cumulated log-performances')
    sub1.set_title('Train set')
    sub1.legend([model_type, 'Baseline'], loc="upper left")
    
    sub2 = fig.add_subplot(2,2,3)
    model.test(val_lags, val_target, n_periods)
    sub2.plot(val_dates, np.cumsum(model.log_returns))
    sub2.plot(val_dates, np.cumsum(val_target.to('cpu')))
    sub2.tick_params(labelrotation=45)
    sub2.grid(True)
    sub2.set_ylabel('Cumulated log-performances')
    sub2.set_title('Validation set')

    sub3 = fig.add_subplot(2,2,4)
    model.test(test_lags, test_target, n_periods)
    sub3.plot(test_dates, np.cumsum(model.log_returns))
    sub3.plot(test_dates, np.cumsum(test_target.to('cpu')))
    sub3.tick_params(labelrotation=45)
    sub3.grid(True)
    sub3.set_title('Test set')
    
    plt.show()


    # Print trading performances on the test set:
    print('Best {} on {} test set achieves:'.format(model_type, dataset))
    print('     Hit-rate: {:.3f}'.format(model.hit_rate))
    print('     Annualized sharpe ratio: {:.3f}'.format(model.sharpe))
    print('     Max drawdown: {:.3f}'.format(model.max_dd))


def load_pretrained_model(model_type, dataset):
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
    
    with open(model_path, "rb") as f:
        model = torch.load(f, map_location=DEVICE)
        model.move_to(DEVICE)
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the best model of a given model type trained on a given dataset.')
    parser.add_argument('model_type', metavar='model_type', type=str, help='An string defining the model type to use.')
    parser.add_argument('dataset', metavar='dataset', type=str, help='A string defining the name of the dataset to use.')
    args = parser.parse_args()
    
    main(args.model_type, args.dataset)