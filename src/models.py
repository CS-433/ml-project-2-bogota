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
    
    def load_pretrained_model(self):
        # This loads the parameters saved in bestmodel.pth into the model
        model_path = '../models/bestmodel.pth'
        m_state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.load_state_dict(m_state_dict)
    
    def train(self, train_input, train_target, num_epochs, mini_batch_size):
            # train_input : tensor of size (N, D) where D is the numbers of lags used to forecast
            # train_target : tensor of size (N)
            
            shuffled_idx = torch.randperm(train_input.size(0))
            train_input = train_input[shuffled_idx,:]
            train_target = train_target[shuffled_idx]
            
            train_losses = []
            for e in tqdm(range(num_epochs)):
                epoch_loss = 0
                
                for b in range(0, int(train_input.size(0)/mini_batch_size), mini_batch_size):
                    output = self(train_input.narrow(0, b, mini_batch_size))
                    loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size))
                    epoch_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                #print("Epoch {}: Loss {}".format(e, epoch_loss))
                train_losses.append(epoch_loss)
            
            return train_losses

    # def predict(self, test_input):
    #     test_output = self(test_input)
    #     return test_output