import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from src.trading_utils import get_log_returns, get_returns, compute_hit_rate, compute_max_drawdown, compute_sharpe


class Model(nn.Module):
    def __init__(self, nb_lags, lr):
        # instantiate model + optimizer + loss function 
        super(Model, self).__init__()
        self.layer1 = nn.Linear(in_features=nb_lags, out_features=6)
        self.layer2 = nn.Linear(in_features=6, out_features=3)
        self.layer3 = nn.Linear(in_features=3, out_features=1)
        
        self.optimizer = optim.Rprop(self.parameters())
        self.criterion = nn.MSELoss(reduction='sum')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  
        
        self.best_hit_rate = 0
    
    
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x.view(-1)
    
    
    def train(self, train_lags, train_target, num_epochs, val_lags=None, val_target=None):
            # train_input : tensor of size (N, D) where D is the numbers of lags used to forecast
            # train_target : tensor of size (N)
            
            # Standardize
            self.mean_ = train_target.mean()
            self.std_ = train_target.std()
            normalized_train_lags = self.standardize(train_lags)
            
            train_losses = []
            val_losses = []
            train_hit_rates = []
            val_hit_rates = []
            
            for e in tqdm(range(num_epochs), desc='Training model', unit=' epoch'):
                output = self(normalized_train_lags)
                loss = self.criterion(output, train_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
                train_hit_rates.append(compute_hit_rate(self, train_lags, train_target))
                
                if (val_lags is not None) and (val_target is not None):
                    val_losses.append(self.criterion(self(val_lags), val_target).item())
                    val_hit_rates.append(compute_hit_rate(self, val_lags, val_target))
                    
                    if val_hit_rates[-1] > self.best_hit_rate:
                        self.best_hit_rate = val_hit_rates[-1]
                        torch.save(self.state_dict(), 'models/bestmodel.pth')
            
            self.train_losses = train_losses
            self.val_losses = val_losses
            self.train_hit_rates = train_hit_rates
            self.val_hit_rates = val_hit_rates
    
    
    def test(self, lags, target):
        self.log_returns = get_log_returns(self, lags, target)
        self.returns = get_returns(self, lags, target)
        self.hit_rate = compute_hit_rate(self, lags, target)
        self.max_dd = compute_max_drawdown(self.log_returns)
        self.sharpe = compute_sharpe(self.log_returns, n_periods=365)
    
    
    def forecast(self, lags):
        return self(self.standardize(lags))
    
    
    def standardize(self, log_returns):
        return (log_returns - self.mean_)/self.std_