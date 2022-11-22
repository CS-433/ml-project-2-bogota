import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
#from pathlib import Path


class Model(nn.Module):
    def __init__(self, nb_lags, lr):
        # instantiate model + optimizer + loss function 
        super().__init__()
        
        self.layer1 = nn.Linear(in_features=nb_lags, out_features=6)
        self.layer2 = nn.Linear(in_features=6, out_features=3)
        self.layer3 = nn.Linear(in_features=3, out_features=1)
        
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  
    
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x.view(-1)
    
    def train(self, train_input, train_target, num_epochs, mini_batch_size):
            # train_input : tensor of size (N, D) where D is the numbers of lags used to forecast
            # train_target : tensor of size (N)
            
            shuffled_idx = torch.randperm(train_input.size(0))
            train_input = train_input[shuffled_idx,:]
            train_target = train_target[shuffled_idx]
            
            train_losses = []
            for e in tqdm(range(num_epochs)):
                epoch_loss = 0
                
                for b in range(0, int(train_input.size(0)/mini_batch_size), 1):
                    output = self(train_input.narrow(0, b, mini_batch_size))
                    loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size))
                    epoch_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                #print("Epoch {}: Loss {}".format(e, epoch_loss))
                train_losses.append(epoch_loss)
            
            self.train_losses = train_losses

    def test(self, val_input, val_target):
        output = self(val_input)
        positive = val_target > 0
        negative = val_target < 0
        print(positive.sum())
        print(negative.sum())
        print(len(val_target), positive.sum() + negative.sum())
        # hit_rate = output[positive_forecast].sign().eq(val_target[positive_forecast].sign()).sum() / positive_forecast.size(0)
        # print(hit_rate)
        # self.hit_rate = hit_rate
        # self.max_draw_down = max_draw_down
        # self.annualized_sharpe = annualized_sharpe
    
    # def predict(self, test_input):
    #     test_output = self(test_input)
    #     return test_output