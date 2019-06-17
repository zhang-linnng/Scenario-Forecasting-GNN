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

# model, GLOW_data_folder, GLOW_data_fname
def generate_point_est(pretrained_model, data_folder, data_fname):
    print('------------------------------------------------------')
    print('Generate point estimates using pretrianed AR model ...')

    save_folder = data_folder
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

    class ARModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(ARModel, self).__init__() 
            self.linear = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            out = self.linear(x)
            return out

    input_dim = 24
    output_dim = 1
    model = ARModel(input_dim,output_dim)

    # Load pre-trained AR model
    #model = torch.load(ARModel_save_folder + ARModel_save_fname)
    model = pretrained_model 
    model.eval()


    # Save AR's point estimates 
    def test(model, test_loader):
        model.eval()
        predictions = []
        true_input_data = []
        for index, data in enumerate(test_loader):
            #if index == 10: break
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
                    true_input_data = input_data
                else:
                    predictions = torch.cat((predictions, pred))
                    true_input_data = torch.cat((true_input_data, input_data))

        return predictions, true_input_data


    pred_on_train, train_data = test(model, train_loader)
    pred_on_valid, valid_data = test(model, valid_loader)
    pred_on_test, test_data = test(model, test_loader)

    np.save(save_folder + 'point_estimates_on_train.npy', pred_on_train)
    np.save(save_folder + 'point_est_training_subset.npy', train_data)

    np.save(save_folder + 'point_estimates_on_valid.npy', pred_on_valid)
    np.save(save_folder + 'point_est_valid_set.npy', valid_data)

    np.save(save_folder + 'point_estimates_on_test.npy', pred_on_test)
    np.save(save_folder + 'point_est_test_set.npy', test_data)
