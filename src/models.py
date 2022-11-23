import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
#from pathlib import Path


class Model(nn.Module):
    def __init__(self, nb_lags, lr):
        # instantiate model + optimizer + loss function 
        super(Model, self).__init__()
        self.layer1 = nn.Linear(in_features=nb_lags, out_features=6)
        self.layer2 = nn.Linear(in_features=6, out_features=3)
        self.layer3 = nn.Linear(in_features=3, out_features=1)
        
        self.optimizer = optim.Rprop(self.parameters(), lr=lr)
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.criterion = nn.MSELoss(reduction='sum')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  
    
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x.view(-1)
    
    def train(self, train_input, train_target, num_epochs, mini_batch_size):
            # train_input : tensor of size (N, D) where D is the numbers of lags used to forecast
            # train_target : tensor of size (N)

            train_losses = []
            lr_history = []
            for e in tqdm(range(num_epochs), desc='Training model', unit=' epoch'):
                #epoch_loss = 0
                # shuffled_idx = torch.randperm(train_input.size(0))
                # train_input = train_input[shuffled_idx,:]
                # train_target = train_target[shuffled_idx]
                
                #for b in range(0, mini_batch_size*int(train_input.size(0)/mini_batch_size), mini_batch_size):
                    #output = self(train_input.narrow(0, b, mini_batch_size))
                    #loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size))
                    #epoch_loss += loss.item()
                    #self.optimizer.zero_grad()
                    #loss.backward()
                    #self.optimizer.step()
                
                output = self(train_input)
                loss = self.criterion(output, train_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #self.scheduler.step(epoch_loss)
                #print("Epoch {}: Loss {}".format(e, epoch_loss))
                train_losses.append(loss.item())
                #lr_history.append(self.get_lr())
            
            self.train_losses = train_losses
            #self.lr_history = lr_history

    def test(self, val_input, val_target):
        output = self(val_input)
        positive_forecast = output > 0

        #rate = output.sign().eq(val_target.sign()).sum() / val_target.size(0)
        self.hit_rate = output[positive_forecast].sign().eq(val_target[positive_forecast].sign()).sum() / positive_forecast.sum()


    #def get_lr(self):
        #for param_group in self.optimizer.param_groups:
            #return param_group['lr']
      
    # def predict(self, test_input):
    #     test_output = self(test_input)
    #     return test_output