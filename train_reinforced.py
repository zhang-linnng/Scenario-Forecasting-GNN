import argparse
import copy
import os 

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

import reinforced_flows as fnn
#import flows as fnn
import load

# GLOW_save_folder, GLOW_data_folder, GLOW_data_fname
def train_flow(save_folder, data_folder, data_fname):
    print('--------------------')
    print('Train reinforced flow ...')
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GLOW')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='input batch size for training (default: 100)')

    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1,
        help='input batch size for testing (default: 1000)')

    parser.add_argument(
        '--epochs',
        type=int,
        default=2000,
        help='number of epochs to train (default: 1000)')

    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate (default: 0.0001)')

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')

    parser.add_argument(
        '--num-blocks',
        type=int,
        default=9,
        help='number of invertible blocks (default: 5)')

    parser.add_argument(
        '--num-hidden',
        type=int,
        default=256,
        help='number of hidden layer neurons')

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
        '--seed', type=int, default=1, help='random seed (default: 1)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")


    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    try:
        os.makedirs(save_folder)
    except OSError:
        pass

    # Load training_subset, valid_set and test_set
    # Just one split: 1-fold
    training_subset, valid_set, test_set = load.load_dataset(data_folder, data_fname)
    print('Training subset size:', training_subset.N)
    print('Validation set size:', valid_set.N)
    print('Test set size:', test_set.N)

    # Load point estimate
    #pred_on_train, pred_on_valid, pred_on_test = load.load_point_estimates(data_folder)

    # Transform to torch.Tensor
    # train_tensor = torch.from_numpy(training_subset.X)
    new_training_subset = np.concatenate((training_subset.y, training_subset.X),-1)
    #new_training_subset = np.concatenate((training_subset.y, pred_on_train),-1)
    mu = new_training_subset.mean()
    std = new_training_subset.std()
    print('Mean of new train set:',mu)
    print('Std of new train set:',std)

    train_tensor = torch.from_numpy((training_subset.X-mu)/std)
    #train_tensor= torch.from_numpy((pred_on_train-mu)/std)
    train_labels = torch.from_numpy((training_subset.y-mu)/std)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    valid_tensor = torch.from_numpy((valid_set.X-mu)/std)
    #valid_tensor = torch.from_numpy((pred_on_valid-mu)/std)
    valid_labels = torch.from_numpy((valid_set.y-mu)/std)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor, valid_labels)

    test_tensor = torch.from_numpy((test_set.X-mu)/std)
    #test_tensor = torch.from_numpy((pred_on_test-mu)/std)
    test_labels = torch.from_numpy((test_set.y-mu)/std)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)

    train_loader = torch.utils.data.DataLoader(
                                        train_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=False, 
                                        **kwargs)

    valid_loader = torch.utils.data.DataLoader(
                                        valid_dataset,
                                        batch_size=args.test_batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        **kwargs)

    test_loader = torch.utils.data.DataLoader(
                                        test_dataset,
                                        batch_size=args.test_batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        **kwargs)

    num_inputs = args.num_inputs
    num_cond_inputs = args.num_cond_inputs
    num_hidden = args.num_hidden 

    def build_model():
        modules = []

        mask = torch.arange(0, num_inputs) % 2
        #mask = torch.ones(num_inputs)
        #mask[round(num_inputs/2):] = 0
        mask = mask.to(device).float()

        # build each modules
        for _ in range(args.num_blocks):
            modules += [
                fnn.ActNorm(num_inputs),
                fnn.LUInvertibleMM(num_inputs),
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu')
                ]
            mask = 1 - mask

        # build model
        model = fnn.FlowSequential(*modules)

        # initialize
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        model.to(device)

        return model

    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    train_loss = []
    def train(epoch):
        model.train()

        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(device)
                else:
                    cond_data = None

                data = data[0]
            data = data.to(device)
            optimizer.zero_grad()
            loss = -model.log_probs(data, cond_data).mean() 
            train_loss.append(loss.item()) 

            loss.backward()
            optimizer.step()

    def validate(epoch, model, loader, prefix='Validation'):
        model.eval()
        val_loss = 0

        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(device)
                else:
                    cond_data = None

                data = data[0]
            data = data.to(device)
            with torch.no_grad():
                val_loss +=  -model.log_probs(data, cond_data).sum().item() 

        return val_loss / len(loader.dataset)

    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model

    valid_loss = []

    for epoch in range(args.epochs):
        print('\nEpoch: {}'.format(epoch))

        train(epoch)
        validation_loss = validate(epoch, model, valid_loader)

        valid_loss.append(validation_loss)

        if epoch - best_validation_epoch >= 30 and epoch > 100:
        #if epoch - best_validation_epoch >= 30:
            break

        if validation_loss < best_validation_loss:
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)

        print(
            'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
            format(best_validation_epoch, -best_validation_loss))

    plt.figure(figsize=(10,10))
    plt.plot(range(len(valid_loss)), valid_loss)
    plt.title('validation loss over epochs')
    plt.savefig(save_folder+'valid_loss.png')

    # Save trained model
    torch.save(best_model, save_folder+'best_model.pt')

    def calculate_dist(true, generated):
        distance_of_one_sample = []
        for t in range(generated.shape[1]):
        
            y = true[t]
            y_hat = generated[:,t]
        
            dist = []
            for p in range(50, 101, 1):
                if p == 50:
                    median = stats.scoreatpercentile(y_hat, p)
                    dist.append(np.abs(y - median))
                else:
                    pl = 100 - p 
                    pu = p
                    l = stats.scoreatpercentile(y_hat, pl)
                    u = stats.scoreatpercentile(y_hat, pu)
                    
                    if y <= u and y >= l:
                        dist.append(0.0)
                    elif y < l:
                        dist.append(np.abs(y - l))
                    else:
                        dist.append(np.abs(y - u))
                        
            dist = np.array(dist)
            if t == 0:
                distance_of_one_sample = dist
            else:
                distance_of_one_sample += dist
                    
        return distance_of_one_sample/24

    def test(model, test_loader):
        model.eval()
        median_pred = []
        ground_truth = []
        point_pred = []
        pi_1 = []
        pi_99 = []
        pi_5 = []
        pi_95 = []
        pi_15 = []
        pi_85 = []
        pi_25 = []
        pi_75 = []
        distance = {}
        for index, data in enumerate(test_loader):
            #if index == 2: break
            inputs = data[0]
            cond_inputs = data[1]

            with torch.no_grad():
                cond_inputs_ = cond_inputs.view(-1,num_cond_inputs) * torch.ones([5000,num_cond_inputs])
                yt_hat =  model.sample(5000, cond_inputs = cond_inputs_).detach().cpu().numpy()
                
                #test_data = test_set.X[index,:].flatten()
                input_data = inputs.detach().numpy().flatten()
                cond_data = cond_inputs.detach().numpy().flatten()
        
                input_data = input_data*std + mu
                cond_data = cond_data*std + mu
                synth = yt_hat*std + mu
            median = stats.scoreatpercentile(synth, 50, axis = 0)
            percentile1 = stats.scoreatpercentile(synth, 1, axis = 0) 
            percentile99 = stats.scoreatpercentile(synth, 99, axis = 0) 
            percentile5 = stats.scoreatpercentile(synth, 5, axis = 0) 
            percentile95 = stats.scoreatpercentile(synth, 95, axis = 0)
            percentile15 = stats.scoreatpercentile(synth, 15, axis = 0) 
            percentile85 = stats.scoreatpercentile(synth, 85, axis = 0)
            percentile25 = stats.scoreatpercentile(synth, 25, axis = 0) 
            percentile75 = stats.scoreatpercentile(synth, 75, axis = 0)
            if index == 0:
                median_pred = median
                ground_truth = input_data
                pi_1 = percentile1
                pi_99 = percentile99
                pi_5 = percentile5
                pi_95 = percentile95
                pi_15 = percentile15
                pi_85 = percentile85
                pi_25 = percentile25
                pi_75 = percentile75
            else:
                median_pred = np.concatenate((median_pred, median))
                ground_truth = np.concatenate((ground_truth, input_data))
                pi_1 = np.concatenate((pi_1, percentile1))
                pi_99 = np.concatenate((pi_99, percentile99))
                pi_5 = np.concatenate((pi_5, percentile5))
                pi_95 = np.concatenate((pi_95, percentile95))
                pi_15 = np.concatenate((pi_15, percentile15))
                pi_85 = np.concatenate((pi_85, percentile85))
                pi_25 = np.concatenate((pi_25, percentile25))
                pi_75 = np.concatenate((pi_75, percentile75))
        
            # distance of test data {index} averaged over 24 hours
            distance[index] = calculate_dist(input_data, synth)

        GLOW_pred_dict = {}
        GLOW_pred_dict['median_pred'] = median_pred
        GLOW_pred_dict['ground_truth'] = ground_truth
        GLOW_pred_dict['pi1'] = pi_1
        GLOW_pred_dict['pi99'] = pi_99
        GLOW_pred_dict['pi5'] = pi_5
        GLOW_pred_dict['pi95'] = pi_95
        GLOW_pred_dict['pi15'] = pi_15
        GLOW_pred_dict['pi85'] = pi_85
        GLOW_pred_dict['pi25'] = pi_25
        GLOW_pred_dict['pi75'] = pi_75
        # Save GLOW_pred_dict as .csv file
        GLOW_pred = pd.DataFrame.from_dict(GLOW_pred_dict)
        GLOW_pred.to_csv(save_folder+'GLOW_pred.csv')


        GLOW_distance = pd.DataFrame.from_dict(distance)
        GLOW_distance.to_csv(save_folder+'GLOW_distance.csv')
        #series = series.mean(axis = 1)

        #GLOW_distance = series.values 
        # Save GLOW_distance as an array
        #np.save(save_folder+'GLOW_distance.npy', GLOW_distance)

        return None

    test(model, test_loader)