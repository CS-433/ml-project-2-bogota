import torch
import time
import matplotlib.pyplot as plt
from models import Model


DATA_FOLDER = '../data/'

if __name__ == '__main__':
    model = Model(nb_lags=6, lr=1e-2)
    
    ######################### Load training data: #########################
    train_input, train_target = torch.load(DATA_FOLDER + 'train_data.pkl')

    ######################### Load validation data: #########################
    val_input, val_target = torch.load(DATA_FOLDER + 'val_data.pkl')

    ######################### Training: #########################
    train = True
    if train:
        print("Training model")
        start = time.time()
        losses = model.train(train_input, train_target, num_epochs=50, mini_batch_size=5)
        torch.save(model.state_dict(), '../models/bestmodel.pth')
        end = time.time()
        print("Elapsed time: {}s".format(end-start))
    
    ######################### Plot training loss: #########################
    plt.plot(losses)
    plt.show()