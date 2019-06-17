import copy
import os 
import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from scipy import linalg as linalg
from scipy.interpolate import interp1d
from scipy import stats

import load
import run_ar

# train_AR(pretrain_data_folder, pretrain_data_fname, ARModel_save_folder, ARModel_save_fname)
def train_AR(data_folder, data_fname, GLOW_data_folder, GLOW_data_fname, model_save_folder):
    # Training settings
    print('----------------------------------')
    print('Pre-train Autoregressive model ...')
    parser = argparse.ArgumentParser(description='PyTorch GLOW')
    parser.add_argument(
        '--epochs',
        type=int,
        default=2000,
        help='number of epochs to train (default: 500)')

    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate (default: 0.0001)')

    parser.add_argument(
        '--num-inputs',
        type=int,
        default=24,
        help='look-ahead horizon of forecasting')

    parser.add_argument(
        '--num-cond-inputs',
        type=int,
        default=24,
        help='length of historical data')

    parser.add_argument(
        '--order',
        type=int,
        default=24,
        help='order of Autoregressive model')

    parser.add_argument(
        '--delta',
        type=int,
        default=1e-4,
        help='stopping criterion')

    args = parser.parse_args()

    #try:
        #os.makedirs(model_save_folder)
    #except OSError:
        #pass

    # Define ARModel class
    class ARModel(nn.Module):

        def __init__(self, input_dim, output_dim):

            super(ARModel, self).__init__() 
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            out = self.linear(x)
            return out

    input_dim = args.num_cond_inputs
    output_dim = 1
    model = ARModel(input_dim,output_dim)

    # Load dataset
    training_subset, valid_set, test_set = load.load_dataset(data_folder, data_fname)
    print('Training subset size:', training_subset.N)
    print('Validation set size:', valid_set.N)
    print('Test set size:', test_set.N)

    # Transform to torch.Tensor
    train_tensor = torch.from_numpy(training_subset.X)
    train_labels = torch.from_numpy(training_subset.y)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    valid_tensor = torch.from_numpy(valid_set.X)
    valid_labels = torch.from_numpy(valid_set.y)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor, valid_labels)

    test_tensor = torch.from_numpy(test_set.X)
    test_labels = torch.from_numpy(test_set.y)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
        
    train_loader = torch.utils.data.DataLoader(
                                            train_dataset, 
                                            batch_size = 1, 
                                            shuffle = False)

    valid_loader = torch.utils.data.DataLoader(
                                            valid_dataset,
                                            batch_size = 1,
                                            shuffle = False,
                                            drop_last = False)

    test_loader = torch.utils.data.DataLoader(
                                            test_dataset,
                                            batch_size = 1,
                                            shuffle = False,
                                            drop_last = False)
    # Define loss function and optimizer
    criterion = nn.MSELoss()# Mean Squared Loss
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr) #Stochastic Gradient Descent

    # Train AR
    train_loss = []
    def train(epoch):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].squeeze()
                else:
                    cond_data = None

                input_data = data[0].squeeze()
            optimizer.zero_grad()

            num_inputs = input_data.shape[0]
            history = cond_data
            pred = []
            for t in range(num_inputs):
                yt_hat = model.forward(history)
                history = torch.cat((history, yt_hat))
                history = history[-24:]
                if t == 0:
                    pred = yt_hat

                else:
                    pred = torch.cat((pred, yt_hat))

            loss = criterion(pred, input_data)
            train_loss.append(loss.detach().item()) 

            loss.backward()
            optimizer.step()


    def validate(epoch, model, valid_loader):
        model.eval()
        val_loss = 0

        for batch_idx, data in enumerate(valid_loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].squeeze()
                else:
                    cond_data = None

                input_data = data[0].squeeze()

            with torch.no_grad():
                history = cond_data
                pred = []
                for t in range(input_data.shape[0]):
                    yt_hat = model.forward(history)
                    history = torch.cat((history, yt_hat))
                    history = history[-24:]
                    if t == 0:
                        pred = yt_hat

                    else:
                        pred = torch.cat((pred, yt_hat))
                val_loss +=  criterion(pred, input_data).detach().item()

        return val_loss / valid_set.N


    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model

    valid_loss = []
    for epoch in range(args.epochs):
        print('\nEpoch: {}'.format(epoch))

        train(epoch)

        validation_loss = validate(epoch, model, valid_loader)
        valid_loss.append(validation_loss)

        if epoch - best_validation_epoch >= 10:
            break

        if validation_loss < best_validation_loss - args.delta:
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)

        print(
            'Best validation at epoch {}: Average mse: {:.4f}'.
            format(best_validation_epoch, best_validation_loss))

    plt.figure(figsize=(10,10))
    plt.plot(range(len(valid_loss)), valid_loss)
    plt.title('validation loss over epochs')
    plt.savefig(model_save_folder+'pretrain_AR_valid_loss.png')


    # Test pre-trained AR
    def test(model, test_loader):
        model.eval()
        predictions = []
        test_data = []
        for index, data in enumerate(test_loader):
            #if index == 2: break
            input_data = data[0].squeeze()
            cond_data = data[1].squeeze()

            with torch.no_grad():
                history = cond_data
                pred = []
                for t in range(input_data.shape[0]):
                    yt_hat = model.forward(history)
                    history = torch.cat((history, yt_hat))
                    history = history[-24:]

                    if t == 0:
                        pred = yt_hat
                    else:
                        pred = torch.cat((pred, yt_hat))

                if index == 0:
                    predictions = pred
                    test_data = input_data
                else:
                    predictions = torch.cat((predictions, pred))
                    test_data = torch.cat((test_data, input_data))

        return predictions, test_data


    predictions, test_data = test(best_model, test_loader)
    # Calculate MSE, plot predictions versus test_data
    print('Pretrain MSE on test data:', criterion(predictions, test_data).detach().item())

    # Save trained model
    #torch.save(best_model, model_save_folder+'best_model.pt')
    torch.save(best_model.state_dict(), model_save_folder+'pretrained_ARmodel.pt')

    # generate point estimates
    run_ar.generate_point_est(best_model, GLOW_data_folder, GLOW_data_fname)
    
    return best_model 


