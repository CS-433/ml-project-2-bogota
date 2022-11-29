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
        
        self.optimizer = optim.Rprop(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='sum')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  
    
    
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x.view(-1)
    
    
    def train(self, train_input, train_target, num_epochs, val_input=None, val_target=None):
            # train_input : tensor of size (N, D) where D is the numbers of lags used to forecast
            # train_target : tensor of size (N)

            train_losses = []
            val_losses = []
            train_hit_rates = []
            val_hit_rates = []
            
            for e in tqdm(range(num_epochs), desc='Training model', unit=' epoch'):
                output = self(train_input)
                loss = self.criterion(output, train_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                
                if (val_input is not None) and (val_target is not None):
                    val_losses.append(self.criterion(self(val_input), val_target).item())
                    train_hit_rates.append(compute_hit_rate(self, train_input, train_target))
                    val_hit_rates.append(compute_hit_rate(self, val_input, val_target))
            
            self.train_losses = train_losses
            self.val_losses = val_losses
            self.train_hit_rates = train_hit_rates
            self.val_hit_rates = val_hit_rates
    
    
    def test(self, val_input, val_target):
        self.log_returns = get_log_returns(self, val_input, val_target)
        self.returns = get_returns(self, val_input, val_target)
        self.hit_rate = compute_hit_rate(self, val_input, val_target)
        self.max_dd = compute_max_drawdown(self.returns)
        self.sharpe = compute_sharpe(self.log_returns, rf=0.01, n=365)