import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
from src.trading_utils import (
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
        self.to(self.device)
        self.criterion = nn.MSELoss()
        self.best_hit_rate = 0

        self.train_losses = []
        self.val_losses = []
        self.train_hit_rates = []
        self.val_hit_rates = []

    def train(
        self,
        train_lags,
        train_target,
        num_epochs,
        batch_size=None,
        val_lags=None,
        val_target=None,
    ):
        # train_input : tensor of size (N, D) where D is the numbers of lags used to forecast
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
                output = self(normalized_train_lags.narrow(0, b, batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, batch_size))
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.train_losses.append(epoch_loss)
            self.train_hit_rates.append(
                compute_hit_rate(self, train_lags, train_target)
            )

            if self.train_hit_rates[-1] > self.best_hit_rate:
                self.best_hit_rate = self.train_hit_rates[-1]
                model_path = Path(
                    __file__
                ).parent.parent / "bestmodels/best{}.pth".format(type(self).__name__)
                torch.save(self.state_dict(), model_path)

            if (val_lags is not None) and (val_target is not None):
                self.val_losses.append(
                    self.criterion(self(val_lags), val_target).item()
                )
                self.val_hit_rates.append(compute_hit_rate(self, val_lags, val_target))

    def test(self, lags, target):
        self.log_returns = get_log_returns(self, lags, target)
        self.returns = get_returns(self, lags, target)
        self.hit_rate = compute_hit_rate(self, lags, target)
        self.max_dd = compute_max_drawdown(self.log_returns)
        self.sharpe = compute_sharpe(self.log_returns, n_periods=365)

    def forecast(self, lags):
        return self(self.standardize(lags))

    def standardize(self, log_returns):
        return (log_returns - self.mean_) / self.std_

    def load_pretrained_model(self):
        # This loads the parameters saved in bestmodels into the model
        model_path = Path(__file__).parent.parent / "bestmodels/best{}.pth".format(
            type(self).__name__
        )
        m_state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.load_state_dict(m_state_dict)


class NN(Model):
    def __init__(self, nb_lags, lr):
        # instantiate model architecture + optimizer
        super().__init__()
        self.layer1 = nn.Linear(in_features=nb_lags, out_features=6)
        self.layer2 = nn.Linear(in_features=6, out_features=3)
        self.layer3 = nn.Linear(in_features=3, out_features=1)

        self.optimizer = optim.Rprop(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x.view(-1)
