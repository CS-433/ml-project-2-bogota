import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from pathlib import Path
from trading_utils import (
    get_log_returns,
    get_returns,
    compute_hit_rate,
    compute_max_drawdown,
    compute_sharpe,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []
        self.train_hit_rates = []
        self.val_hit_rates = []

    def train(
        self,
        train_lags,
        train_target,
        num_epochs,
        val_lags,
        val_target,
        batch_size=None
    ):
        # train_lags : tensor of size (N, D) where D is the numbers of lags used to forecast
        # train_target : tensor of size (N)

        # Standardize
        self.mean_ = train_target.mean()
        self.std_ = train_target.std()
        normalized_train_lags = self.standardize(train_lags)

        if batch_size is None:
            batch_size = train_lags.shape[0]

        for e in tqdm(range(num_epochs), desc="Training model", unit=" epoch"):
            epoch_loss = 0

            for b in range(
                0, train_lags.shape[0] - train_lags.shape[0] % batch_size, batch_size
            ):    
                # Forward pass:
                output = self(normalized_train_lags.narrow(0, b, batch_size))
                
                # Loss computation:
                loss = self.criterion(output, train_target.narrow(0, b, batch_size))
                epoch_loss += loss.item()

                # Backward pass:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.train_losses.append(epoch_loss)
            self.train_hit_rates.append(
                compute_hit_rate(self, train_lags, train_target)
            )
            self.val_losses.append(
                self.criterion(self(val_lags), val_target).item()
            )
            self.val_hit_rates.append(compute_hit_rate(self, val_lags, val_target))

    def test(self, lags, target, n_periods):
        self.log_returns = get_log_returns(self, lags, target)
        self.returns = get_returns(self, lags, target)
        self.hit_rate = compute_hit_rate(self, lags, target)
        self.max_dd = compute_max_drawdown(self.returns)
        self.sharpe = compute_sharpe(self.log_returns, n_periods=n_periods)

    def forecast(self, lags):
        return self(self.standardize(lags))

    def standardize(self, log_returns):
        return (log_returns - self.mean_) / self.std_


class NN(Model):
    def __init__(self, nb_lags, lr):
        # instantiate model architecture + optimizer
        super().__init__()
        self.fc1 = nn.Linear(in_features=nb_lags, out_features=6)
        self.fc2 = nn.Linear(in_features=6, out_features=3)
        self.fc3 = nn.Linear(in_features=3, out_features=1)

        self.optimizer = optim.Rprop(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class CNN(Model):
    def __init__(self, nb_lags, lr):
        # instantiate model architecture + optimizer
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=4*(nb_lags-(3-1)), out_features=1)
        
        self.optimizer = optim.Rprop(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x[:,None,:])
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x.view(-1)


class LSTM(Model):
    def __init__(self, nb_lags, lr):
        # instantiate model architecture + optimizer
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = 1,
            hidden_size = 2,
            num_layers = 2,
            batch_first = True,
            dropout = 0
        )
        self.fc = nn.Linear(in_features=2, out_features=1)
        
        self.optimizer = optim.Rprop(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x[:,:,None])
        out = self.fc(hidden[-1])
        return out.view(-1)


class RandomForest(RandomForestRegressor):
    def __init__(self, n_estimators, max_features, random_state):
        super().__init__(n_estimators=n_estimators, max_features=max_features, random_state=random_state)
    
    def train(self, train_lags, train_target):
        # train_lags : tensor of size (N, D) where D is the numbers of lags used to forecast
        # train_target : tensor of size (N)
        print('Training Random Forest with n_estimators={} and max_features={:.2f} ...'.format(self.n_estimators, self.max_features))
        self.fit(train_lags, train_target)

    def test(self, lags, target, n_periods):
        self.log_returns = get_log_returns(self, lags, target)
        self.returns = get_returns(self, lags, target)
        self.hit_rate = compute_hit_rate(self, lags, target)
        self.max_dd = compute_max_drawdown(self.returns)
        self.sharpe = compute_sharpe(self.log_returns, n_periods=n_periods)

    def forecast(self, lags):
        return self.predict(lags)


class Ensemble:
    def __init__(self, model_type, nb_models, nb_lags, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == NN:
            self.model_type_str = 'NN'
        elif model_type == CNN:
            self.model_type_str = 'CNN'
        elif model_type == LSTM:
            self.model_type_str = 'LSTM'

        self.nb_models = nb_models
        self.nb_lags = nb_lags
        self.lr = lr

        self.models = []
        for i in range(nb_models):
            torch.manual_seed(i)
            self.models.append(model_type(nb_lags=nb_lags, lr=lr))

    def train(self, train_lags, train_target, num_epochs, val_lags, val_target, batch_size=None):
        # train_lags : tensor of size (N, D) where D is the numbers of lags used to forecast
        # train_target : tensor of size (N)

        print('Training ensemble of {} {} models with lr = {:.4f}:'.format(self.nb_models, self.model_type_str, self.lr))
        for i in range(self.nb_models):
            self.models[i].train(train_lags, train_target, num_epochs=num_epochs, val_lags=val_lags, val_target=val_target, batch_size=batch_size)
        print()
    
    def test(self, lags, target, n_periods):
        self.log_returns = get_log_returns(self, lags, target)
        self.returns = get_returns(self, lags, target)
        self.hit_rate = compute_hit_rate(self, lags, target)
        self.max_dd = compute_max_drawdown(self.returns)
        self.sharpe = compute_sharpe(self.log_returns, n_periods=n_periods)

    def forecast(self, lags):
        median_forecast = torch.zeros((lags.shape[0], self.nb_models), device=self.device)
        for i in range(self.nb_models):
            median_forecast[:,i] = self.models[i].forecast(lags)
        median_forecast = median_forecast.median(dim=1)
        return median_forecast.values