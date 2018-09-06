from __future__ import division
import multiprocessing
import argparse

from keras import regularizers

# Local packages.
import network
import dataset
import models

DEFAULT_EPOCHS = 400
DEFAULT_REPETITIONS = 1

params = {
    'activation': 'tanh',
    'verbose': 0,
    'debug': 0,
    'batch_size': 500000,
    'fc_hidden_u': 20,
    'dropout': 0.2,
    'lr': 0.001,
}

# For hardcoded context layers
params['cols'] = {
    'activity': [0, 1, 2, 5, 7, 8, 9, 28, 30, 32, 34, 36, 41, 43, 45],
    'peers': [3, 6, 10, 29, 31, 33, 35, 37, 42, 44, 46],
    'courseinfo': [4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                   25, 26, 27, 38],
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
        'activityreg': None,
    }
}


def get_args_parser():

    parser = argparse.ArgumentParser(description='Specify the test file')
    parser.add_argument('--run-prefix', dest='run_prefix')
    parser.add_argument('--test-datasets', dest='test_datasets')
    parser.add_argument('--model-names', dest='model_names')
    parser.add_argument('--processes', dest='processes', default=1, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        default=DEFAULT_EPOCHS, type=int)
    parser.add_argument('--repetitions', dest='repetitions',
                        default=DEFAULT_REPETITIONS, type=int)
    return parser


parser = get_args_parser()
args = parser.parse_args()

params['repetitions'] = args.repetitions
params['epochs'] = args.epochs

networks = network.get_combinations(args, models.get_all())

datasets, params = dataset.load(params, args.test_datasets)

manager = multiprocessing.Manager()
model_scores = manager.dict()

nxt = 0
while nxt < len(networks):

    # Just references, no duplicated in memory.
    processes = []
    for i in range(nxt, nxt + args.processes):

        if i >= len(networks):
            # No more networks to process.
            break

        p = multiprocessing.Process(target=network.test, args=(
            networks[i], datasets, params, args.run_prefix, model_scores))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    nxt = nxt + args.processes

by_dataset = {}
for model_score in model_scores.values():
    if model_score['dataset'] not in by_dataset:
        by_dataset[model_score['dataset']] = {}
    by_dataset[model_score['dataset']][model_score['name']] = [
        model_score['f1'],
        model_score['acc'],
        model_score['recall'],
    ]

for dataset_id, models_data in by_dataset.items():
    print('Dataset ' + dataset_id)
    for model_name, result in models_data.items():
        print(model_name + ',' +
              str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]))

