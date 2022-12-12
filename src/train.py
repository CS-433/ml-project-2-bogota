import torch
import time
import matplotlib.pyplot as plt
from models import Model


DATA_FOLDER = "../data/"

if __name__ == "__main__":
    model = Model(nb_lags=6, lr=1e-2)

    ######################### Load training data: #########################
    train_input, train_target = torch.load(DATA_FOLDER + "train_data.pkl")

    ######################### Load validation data: #########################
    val_input, val_target = torch.load(DATA_FOLDER + "val_data.pkl")

    ######################### Training: #########################
    train = False
    if train:
        print("Training model")
        start = time.time()
        losses = model.train(
            train_input, train_target, num_epochs=100, mini_batch_size=5
        )
        torch.save(model, "../models/bestmodel.pth")
        end = time.time()
        print("Elapsed time: {}s".format(end - start))

    ######################### Validation: #########################
    model = torch.load("../models/bestmodel.pth")
    model.test(val_input, val_target)

    ######################### Plot training loss: #########################
    plt.plot(model.train_losses)
    plt.show()
