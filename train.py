import torch
import matplotlib.pyplot as plt
from src.models import NN

DATA_FOLDER = 'data/'

if __name__ == "__main__":
    model = NN(nb_lags=6, lr=1e-2)

    ######################### Load training data: #########################
    train_lags, train_target = torch.load(DATA_FOLDER + 'train_data.pkl')
    train_lags, train_target = train_lags.to(model.device), train_target.to(model.device)

    ######################### Load validation data: #########################
    val_lags, val_target = torch.load(DATA_FOLDER + 'val_data.pkl')
    val_lags, val_target = val_lags.to(model.device), val_target.to(model.device)

    ######################### Training: #########################
    train = True
    if train:
        model.train(train_lags, train_target, num_epochs=1000, val_lags=val_lags, val_target=val_target)

    ######################### Validation: #########################
    model.load_pretrained_model()
    model.test(val_lags, val_target)

    ######################### Plot losses: #########################
    plt.plot(model.train_losses)
    plt.plot(model.val_losses)
    plt.show()