import numpy as np
import pandas as pd 
import os

def format(df):
    data = df['use'].values

    #valid_N = round(round(data.shape[0]/24)*24)
    #data = data[:valid_N]

    data = data.reshape(-1,24)
    X = data[1:,:]
    N_tilde = X.shape[0]
    y = data[:N_tilde,:]
    data = np.concatenate((X,y),1)

    return data

def get_train_mean_and_std(df):
    data = df['use'].values
    mean = data.mean()
    std = data.std()

    return mean, std

def read_csv_and_split(data_folder, data_fname):
    #folder = 'datasets/aggregation-level-10/'
    #file_name = file_name +'-{}.csv'.format(trial)
    if data_fname == "Aggregated-2013-2014.csv":
        # load pre-train AR data
        test_start = '2014-11-01 06:00:00'
        valid_start = '2014-09-01 06:00:00'

    elif data_fname == "Aggregated-2015-2017.csv":
        # load train GLOW data
        test_start = '2017-10-01 06:00:00'
        valid_start = '2017-07-01 06:00:00'

    elif data_fname == "Aggregated-2013-2017.csv":
        # load train AR data
        test_start = '2017-10-01 06:00:00'
        valid_start = '2017-07-01 06:00:00'
    else:

        print('Unknown file name!')

    #print('test_start:')
    #print(test_start)
    #print('valid_start:')
    #print(valid_start)
    try:
        df = pd.read_csv(data_folder+data_fname, 
                    infer_datetime_format=True, 
                    parse_dates={'datetime':[0]},
                    index_col=['datetime'])
    except:
        df = pd.read_csv(data_folder+data_fname, 
                    index_col=['datetime'])

    test_mask = df.index >= test_start
    test_df = df[test_mask]
    df = df[df.index < test_start]
    valid_mask = df.index >= valid_start
    valid_df = df[valid_mask]
    train_df = df[df.index < valid_start]

    print('test_df:')
    print(test_df.info())
    print('valid_df:')
    print(valid_df.info())
    print('train_df:')
    print(train_df.info())

    #train_mean, train_std = get_train_mean_and_std(train_df)

    training_subset = format(train_df)
    valid_set = format(valid_df)
    test_set = format(test_df)

    return training_subset, valid_set, test_set



def load_dataset(data_folder, data_fname):
    # load pre-train dataset 
    # ---------------------- or GLOW training dataset 
    # ---------------------- or AR training dataset 
    training_subset, valid_set, test_set = read_csv_and_split(data_folder, data_fname)

    class Dataset:
        def __init__(self, data, num_inputs):
            self.X = data[:,:num_inputs].astype(np.float32)
            self.y = data[:,num_inputs:].astype(np.float32)
            self.N = self.X.shape[0]

    num_inputs = 24
    training_subset = Dataset(training_subset, num_inputs) # return a Dataset class
    valid_set = Dataset(valid_set, num_inputs)
    test_set = Dataset(test_set, num_inputs)

    return training_subset, valid_set, test_set


def load_point_estimates(data_folder):
    pred_on_train = np.load(data_folder+'point_estimates_on_train.npy')

    pred_on_valid = np.load(data_folder+'point_estimates_on_valid.npy')

    pred_on_test = np.load(data_folder+'point_estimates_on_test.npy')


    pred_on_train = pred_on_train.reshape(-1,24)
    pred_on_valid = pred_on_valid.reshape(-1,24)
    pred_on_test = pred_on_test.reshape(-1,24)

    return pred_on_train, pred_on_valid, pred_on_test

