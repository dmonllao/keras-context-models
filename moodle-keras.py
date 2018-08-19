from __future__ import division

import os
import re
import sys
import time
import csv
import argparse
from datetime import datetime

# I don't want the Using xxx garbage.
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

from keras import regularizers
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras import backend as K

import numpy as np
#np.random.seed(137)

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
#from tensorflow import set_random_seed
#set_random_seed(2)

import pandas as pd
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

# Local packages.
import layer
import network
import debugger

#########################################################################################

params = {
    'activation': 'tanh',
    'verbose': 0,
    'batch_size': 500000,
    'epochs': 400,
    'repetitions': 3,
    'fc_hidden_u': 20,
    'dropout': 0.2,
    'lr': 0.001,
}

# For hardcoded context layers
params['cols'] = {
    'activity': [0, 1, 2, 5, 7, 8, 9, 28, 30, 32, 34, 36, 39],
    'peers': [3, 6, 10, 29, 31, 33, 35, 37, 40],
    'courseinfo': [4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 38],
}
params['cols']['ctx'] = params['cols']['peers'] + params['cols']['courseinfo']

# For separated layers layers.
params['separate_cols'] = {
    'activity': {
        'cols': params['cols']['activity'],
        'units': 7,
        'activation': 'tanh',
        'kernelreg': None,
        'activityreg': None,
    },
    'peers': {
        'cols': params['cols']['peers'],
        'units': 2,
        'activation': 'tanh',
        'kernelreg': None,
        'activityreg': None,
    },
    'courseinfo': {
        'cols': params['cols']['courseinfo'],
        'units': 2,
        'activation': 'relu',
        'kernelreg': regularizers.l2(0.01),
        'activityreg': regularizers.l1(0.01),
    }
}

def get_training_samples(train_files):

    for train_file in train_files:
        train = pd.read_csv(train_file, skiprows=3, dtype=np.float32)

        try:
            x_train = np.concatenate([x_train, train[train.columns[:-1]].fillna(0)])
        except NameError:
            x_train = train[train.columns[:-1]].fillna(0)

        try:
            y_train = np.concatenate([y_train, np.eye(2)[train[train.columns[-1]].astype(int)]])
        except NameError:
            y_train = np.eye(2)[train[train.columns[-1]].astype(int)]

    return x_train, y_train

def get_testing_samples(test_file):
    test = pd.read_csv(test_file, skiprows=3, dtype=np.float32)
    x_test = test[test.columns[:-1]].fillna(0)
    y_test = np.eye(2)[test[test.columns[-1]].astype(int)]

    return x_test, y_test

def get_args_parser():

    parser = argparse.ArgumentParser(description='Specify the test file')
    parser.add_argument('--run-prefix', dest='run_prefix')
    parser.add_argument('--test-dataset', dest='test_dataset')
    parser.add_argument('--model-names', dest='model_names')
    return parser

def get_datasets(test_dataset_id):

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Loading in memory. Not a massive issue as these datasets are relatively small.
    test_file = os.path.join(script_dir, 'datasets', test_dataset_id + '.csv')
    train_files = []
    for f in os.listdir(os.path.join(script_dir, 'datasets')):
        if f == test_dataset_id + '.csv':
            continue
        if re.match('^dataset\d\.csv', f) is None:
            continue

        train_files.append(os.path.join(script_dir, 'datasets', f))

    data = {'test_dataset_id': test_dataset_id}

    data['x_train'], data['y_train'] = get_training_samples(train_files)
    data['x_test'], data['y_test'] = get_testing_samples(test_file)
    data['n_classes'] = data['y_train'].shape[1]
    data['n_features'] = data['x_train'].shape[1]

    #print('Testing ' + test_dataset_id + ' with ' + str(len(train_files)) + ' training datasets')
    #print('  Data train size: '+ str(data['x_train'].shape[0]))
    #print('  Data test size: ' + str(data['x_test'].shape[0]))
    #print('  Total num features: ' + str(data['n_features']))

    return data

def list_dataset_ids():
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    return sorted([os.path.splitext(x)[0] for x in os.listdir(datasets_dir) if re.match('^dataset\d\.csv', x)])

def get_networks(args, model_data):

    # List of datasets we will use.
    if args.test_dataset is not None:
        test_dataset_list = [args.test_dataset]
    else:
        # All available datasets.
        test_dataset_list = list_dataset_ids()

    # List of models we will evaluate.
    if args.model_names is not None:
        names = args.model_names.split(',')
        test_models = []
        for model_data in models:
            for model_name in names:
                if model_data['name'] == model_name:
                    test_models.append(model_data)
                    break
    else:
        # All existing models.
        test_models = models

    networks = []
    for dataset_id in test_dataset_list:
        for model_data in test_models:
            network = model_data.copy()
            network['dataset'] = dataset_id
            networks.append(network)

    return networks

def test_model(index, model, params, data, name=None):

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=params['lr']),
                  metrics=['accuracy'])

    if params['verbose'] == 1 and index == 0:
        # Print summaries.
        model.summary()

    summary_name = data['test_dataset_id'] + '-' + str(time.time()) + '-' + params['name']
    if name != None:
        summary_name = name + '-' + summary_name

    callbacks = []
    summaries = TensorBoard(log_dir='./summaries/' + summary_name, histogram_freq=0,
        batch_size=params['batch_size'], write_grads=True)
    callbacks.append(summaries)

    metrics = debugger.Metrics(data['x_test'], data['y_test'], summaries)
    callbacks.append(metrics)

    model.fit(data['x_train'], data['y_train'],
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                verbose=params['verbose'],
                validation_split=0,
                callbacks=callbacks,
                shuffle=True)

    print(metrics.scores)
    return metrics.scores

###############################################################################

parser = get_args_parser()
args = parser.parse_args()

models = []

models.append({
    'name': 'No context NN - 1 hidden.',
    'network': network.fc_1h,
    'feature_set': 'nocontext'
})

models.append({
    'name': 'With peers NN - 1 hidden.',
    'network': network.fc_1h,
    'feature_set': 'withpeers'
})

models.append({
    'name': 'All features NN - 1 hidden.',
    'network': network.fc_1h,
    'feature_set': 'all'
})

models.append({
    'name': 'No context NN - 2 hidden.',
    'network': network.fc_2h,
    'feature_set': 'nocontext'
})

models.append({
    'name': 'With peers NN - 2 hidden.',
    'network': network.fc_2h,
    'feature_set': 'withpeers'
})

models.append({
    'name': 'All features NN - 2 hidden.',
    'network': network.fc_2h,
    'feature_set': 'all'
})

models.append({
    'name': 'Activity features trained with context.',
    'network': network.inctx,
    'feature_set': 'all'
})

models.append({
    'name': 'Activity features trained with context (REG).',
    'network': network.inctx_reg,
    'feature_set': 'all'
})

models.append({
    'name': 'Activity features trained with context + original activity features.',
    'network': network.inctx_extra,
    'feature_set': 'all'
})

models.append({
    'name': 'Activity features trained with context + original activity features (REG).',
    'network': network.inctx_extra_reg,
    'feature_set': 'all'
})

models.append({
    'name': 'Separate no-context / 1 context inputs.',
    'network': network.simple_separate_1,
    'feature_set': 'all'
})

models.append({
    'name': 'Separate no-context / 1 context inputs (REG).',
    'network': network.simple_separate_1_reg,
    'feature_set': 'all'
})

models.append({
    'name': 'Separate no-context / all context inputs.',
    'network': network.simple_separate_all,
    'feature_set': 'all'
})

models.append({
    'name': 'Separate no-context / all context inputs (REG).',
    'network': network.simple_separate_all_reg,
    'feature_set': 'all'
})

models.append({
    'name': 'Separate activity / course peers / required.',
    'network': network.complex_separate,
    'feature_set': 'all'
})

#models.append({
    #'name': 'Combinations of features in pairs.',
    #'network': network.comb,
    #'feature_set': 'all'
#})

#models.append({
    #'name': 'Combinations of features in pairs (REG).',
    #'network': network.comb_reg,
    #'feature_set': 'all'
#})

networks = get_networks(args, models)

loaded_test_dataset = None
model_scores = {}

for network in networks:
    start_time = datetime.now()

    if loaded_test_dataset != network['dataset']:
        # No need to reload the datasets if they were already loaded.
        data = get_datasets(network['dataset'])
        loaded_test_dataset = network['dataset']

    print('\nModel ' + network['name'] + ' using ' + network['dataset'] + ' as test dataset')

    acc = []
    f1 = []
    for index in range(params['repetitions']):

        params['n_features'] = data['n_features']
        params['n_classes'] = data['n_classes']
        params['name'] = network['name']
        params['feature_set'] = network['feature_set']

        # Build the network
        inputs, output = network['network'](params)

        model = Model(inputs, output)

        #if index == 0:
            #print('  Total params: ' + str(model.count_params()))

        score = test_model(index, model, params, data, name=args.run_prefix)
        acc.append(score['acc'])
        f1.append(score['f1'])

    if data['test_dataset_id'] not in model_scores.keys():
        # Initialise it if this is this dataset first model
        model_scores[data['test_dataset_id']] = []

    model_score = {
        'name': network['name'],
        'acc': np.average(acc),
        'f1': np.average(f1),
    }
    model_scores[data['test_dataset_id']].append(model_score)

    print('  Time: ' + str(datetime.now() - start_time))
    #print('  Median accuracy: ' + str(np.median(acc)))
    #print('  Average accuracy: ' + str(model_score['acc']))
    #print('  Median F1: ' + str(np.median(f1)))
    #print(' *Average F1: ' + str(model_score['f1']))

for test_dataset_id, model_scores in model_scores.items():
    print("Results for " + test_dataset_id)
    for model_score in model_scores:
        print(model_score['name'] + ', ' + str(model_score['f1']) + ', ' + str(model_score['acc']))
