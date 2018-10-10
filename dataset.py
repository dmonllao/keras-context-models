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
            x_train = np.concatenate(
                [x_train, train[train.columns[:-1]].fillna(0).values])
        except NameError:
            x_train = train[train.columns[:-1]].fillna(0).values

        try:
            y_train = np.concatenate(
                [y_train, np.eye(2)[train[train.columns[-1]].astype(int)]])
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
    for f in os.listdir(os.path.join(script_dir, 'datasets')).sort():
        if re.match('^dataset\d\.csv', f) is None:
            continue
        dataset_id, _ = os.path.splitext(f)
        datasets.append(dataset_id)

    return iter(datasets)


def standardize_activity_and_peers(x, params):

    for col_index in params['cols']['peers']:
        dataset_avg = np.average(x[:, col_index - 1])
        course_meanized = (x[:, col_index - 1] - x[:, col_index]) / 2
        dataset_meanized = (x[:, col_index - 1] - dataset_avg) / 2

        dataset_avg_col = np.full((x.shape[0], 1), dataset_avg)
        dataset_meanized_col = dataset_meanized.reshape(-1, 1)
        course_meanized_col = course_meanized.reshape(-1, 1)
        x = np.concatenate(
            (x, course_meanized_col, dataset_meanized_col, dataset_avg_col),
            axis=1)

    return x


def load(params, test_datasets):

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Loading in memory. Not a massive issue as these datasets
    # are relatively small.
    datasets = {}
    for dataset_id in test_dataset_list(test_datasets):

        # Init the container for all other datasets as training data.
        datasets[dataset_id] = {'test_dataset_id': dataset_id}

        test_file = os.path.join(script_dir, 'datasets', dataset_id + '.csv')
        train_files = []
        for train_dataset_id in it_them():
            if train_dataset_id == dataset_id:
                continue

            train_file = os.path.join(
                script_dir, 'datasets', train_dataset_id + '.csv')
            train_files.append(train_file)

            # Init the container for each other training dataset as
            # training data.
            dataset_key = dataset_id + '-' + train_dataset_id
            datasets[dataset_key] = {'test_dataset_id': dataset_id}
            datasets[dataset_key]['x_train'], datasets[dataset_key]['y_train'] = \
                get_training_samples([train_file])
            datasets[dataset_key]['x_test'], datasets[dataset_key]['y_test'] = \
                get_testing_samples(test_file)

            # Standardize values.
            datasets[dataset_key]['x_test'] = standardize_activity_and_peers(
                datasets[dataset_key]['x_test'], params)
            datasets[dataset_key]['x_train'] = standardize_activity_and_peers(
                datasets[dataset_key]['x_train'], params)

        # Including all other datasets as training data.
        datasets[dataset_id]['x_train'], datasets[dataset_id]['y_train'] = \
            get_training_samples(train_files)
        datasets[dataset_id]['x_test'], datasets[dataset_id]['y_test'] = \
            get_testing_samples(test_file)

        # Standardize values.
        datasets[dataset_id]['x_test'] = standardize_activity_and_peers(
            datasets[dataset_id]['x_test'], params)
        datasets[dataset_id]['x_train'] = standardize_activity_and_peers(
            datasets[dataset_id]['x_train'], params)

    # Add standardize columns as peers cols.
    # It does not matter which dataset we fit in as all of them have the same
    # number of columns.
    params = add_additional_cols(datasets[dataset_id]['x_test'], params)

    # Iterate again now that we have the final number of features.
    for dataset_id in test_dataset_list(test_datasets):
        datasets[dataset_id]['n_classes'] = datasets[dataset_id]['y_train'].shape[1]
        datasets[dataset_id]['n_features'] = datasets[dataset_id]['x_train'].shape[1]

        # Now for each test single-dataset-for-training combination
        for train_dataset_id in it_them():
            if train_dataset_id == dataset_id:
                continue

            dataset_key = dataset_id + '-' + train_dataset_id
            datasets[dataset_key]['n_classes'] = datasets[dataset_key]['y_train'].shape[1]
            datasets[dataset_key]['n_features'] = datasets[dataset_key]['x_train'].shape[1]

    return datasets, params


def add_additional_cols(x, params):

    prev_last = params['cols']['peers'][-1]
    new_len = x.shape[1]

    for i in range(prev_last + 1, new_len):
        params['cols']['peers'].append(i)
        params['cols']['ctx'].append(i)

    return params


def test_dataset_list(test_datasets):

    # List of datasets we will use.
    if test_datasets is not None:
        test_dataset_list = test_datasets.split(',')
    else:
        # All available datasets.
        test_dataset_list = list_ids()

    return test_dataset_list


def list_ids():
    datasets_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'datasets')
    files = os.listdir(datasets_dir)
    return sorted(
        [os.path.splitext(x)[0] for x in files if re.match('^dataset\d\.csv', x)])
