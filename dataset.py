import os
import re

import numpy as np

import pandas as pd
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

def get_training_samples(train_files):

    for train_file in train_files:
        train = pd.read_csv(train_file, skiprows=3, dtype=np.float32)

        try:
            x_train = np.concatenate([x_train, train[train.columns[:-1]].fillna(0).values])
        except NameError:
            x_train = train[train.columns[:-1]].fillna(0).values

        try:
            y_train = np.concatenate([y_train, np.eye(2)[train[train.columns[-1]].astype(int)]])
        except NameError:
            y_train = np.eye(2)[train[train.columns[-1]].astype(int)]

    return x_train, y_train

def get_testing_samples(test_file):
    test = pd.read_csv(test_file, skiprows=3, dtype=np.float32)
    x_test = test[test.columns[:-1]].fillna(0).values
    y_test = np.eye(2)[test[test.columns[-1]].astype(int)]

    return x_test, y_test

def it_them():

    datasets = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(os.path.join(script_dir, 'datasets')):
        if re.match('^dataset\d\.csv', f) is None:
            continue
        dataset_id, _ = os.path.splitext(f)
        datasets.append(dataset_id)

    return iter(datasets)

def standardize_activity_and_peers(x, params):
    #print('yeah')
    #print(x[0])
    #print(x[4])
    for col_index in params['cols']['peers']:
        meanized = (x[:, col_index - 1] - x[:, col_index]) / 2
        #x[:, col_index - 1] = meanized
        x[:, col_index] = meanized
    #print(x[3])
    #print(x[4])
    return x

def load(params, test_datasets):

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Loading in memory. Not a massive issue as these datasets are relatively small.
    datasets = {}
    for dataset_id in test_dataset_list(test_datasets):
        datasets[dataset_id] = {'test_dataset_id': dataset_id}

        test_file = os.path.join(script_dir, 'datasets', dataset_id + '.csv')
        train_files = []
        for train_dataset_id in it_them():
            if train_dataset_id == dataset_id:
                continue

            train_files.append(os.path.join(script_dir, 'datasets', train_dataset_id + '.csv'))

        datasets[dataset_id]['x_train'], datasets[dataset_id]['y_train'] = get_training_samples(train_files)
        datasets[dataset_id]['x_test'], datasets[dataset_id]['y_test'] = get_testing_samples(test_file)
        datasets[dataset_id]['n_classes'] = datasets[dataset_id]['y_train'].shape[1]
        datasets[dataset_id]['n_features'] = datasets[dataset_id]['x_train'].shape[1]

        datasets[dataset_id]['x_train'] = standardize_activity_and_peers(datasets[dataset_id]['x_train'], params)
        datasets[dataset_id]['x_test'] = standardize_activity_and_peers(datasets[dataset_id]['x_test'], params)

        #print('Testing ' + dataset_id + ' with ' + str(len(train_files)) + ' training datasets')
        #print('  Data train size: '+ str(datasets[dataset_id]['x_train'].shape[0]))
        #print('  Data test size: ' + str(datasets[dataset_id]['x_test'].shape[0]))
        #print('  Total num features: ' + str(datasets[dataset_id]['n_features']))

    return datasets

def test_dataset_list(test_datasets):

    # List of datasets we will use.
    if test_datasets is not None:
        test_dataset_list = test_datasets.split(',')
    else:
        # All available datasets.
        test_dataset_list = dataset.list_ids()

    return test_dataset_list

def list_ids():
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    return sorted([os.path.splitext(x)[0] for x in os.listdir(datasets_dir) if re.match('^dataset\d\.csv', x)])
