from __future__ import division
import multiprocessing
import argparse
import numpy as np

from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras import Model, Sequential
from keras.layers import Dense
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

# Local packages.
import network
import dataset
import models
import metric

DEFAULT_EPOCHS = 400
DEFAULT_REPETITIONS = 1

params = {
    'activation': 'tanh',
    'verbose': 0,
    'debug': 0,
    'batch_size': 50000,
    'fc_hidden_u': 20,
    'dropout': 0.2,
    'lr': 0.001,
}

# for hardcoded context layers
params['cols'] = {
    'activity': [0, 1, 2, 5, 7, 8, 9, 28, 30, 32, 34, 36, 41, 43, 45],
    'peers': [3, 6, 10, 29, 31, 33, 35, 37, 42, 44, 46],
    'courseinfo': [4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                   25, 26, 27, 38],
}
params['cols']['ctx'] = params['cols']['peers'] + params['cols']['courseinfo']

# for separated layers layers.
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
        'kernelreg': None,
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

# print('\n\n====== NN ======')

# model_results = []
# for nn in networks:
#     model_results.append(network.test(nn, datasets, params, args.run_prefix))

# by_dataset = {}
# for model_result in model_results:
#     if model_result['dataset'] not in by_dataset:
#         by_dataset[model_result['dataset']] = {}
#     by_dataset[model_result['dataset']][model_result['name']] = model_result

# for dataset_id, models_data in by_dataset.items():
#     print('dataset ' + dataset_id)
#     for model_name, result in models_data.items():
#         print('\n' + model_name)
#         print('Accuracy: ' + str(result['acc']))
#         print('F1 score: ' + str(result['f1']))
#         print('Recall: ' + str(result['recall']))

print('\n\n====== scikit-learn ensembles ======')

def build_model(nn=False,
                params=False,
                options=False):
    """Builds a model using the provided parameters."""

    inputs, output = nn['network'](params, options)
    model = Model(inputs, output)
    model = network.compile_model(model, params)
    return model

for nn in networks:

    print("\n" + nn['name'])
    data = datasets[nn['dataset']]

    network_params, network_options = network.parse_network_obj(
        nn, params, data['n_features'], data['n_classes'])
    kwargs = network.get_fit_kwargs(network_params)

    kwargs['build_fn'] = build_model
    kwargs['nn'] = nn
    kwargs['options'] = network_options
    # Attach the original list of global params as well.
    kwargs['params'] = network_params

    model = KerasClassifier(**kwargs)
    # ensemble = BaggingClassifier(model, n_estimators=10, max_samples=0.2)
    ensemble = AdaBoostClassifier(model, n_estimators=10)

    y_train_1d = np.argmax(data['y_train'], axis=1)
    ensemble.fit(data['x_train'], y_train_1d)

    y_pred_labels_1d = ensemble.predict(data['x_test'])

    acc, f1, recall = metric.get(y_pred_labels_1d, data['y_test'])

    print('Accuracy: ' + str(acc))
    print('F1 score: ' + str(f1))
    print('Recall: ' + str(recall))

print('\n\n====== david ensembles ======')

# networks = network.get_separate_training_datasets_combinations(args, models.get_all())
# models = []
# for dataset_id, dataset_networks in networks.items():
#     print('Dataset: ' + dataset_id)

#     pred_labels = []

#     for nn in dataset_networks:
#         results = network.test(nn, datasets, params, args.run_prefix)

#         model = results['model']
#         data = datasets[nn['dataset']]

#         y_pred = model.predict(data['x_test'])

#         # Output activations.
#         try:
#             pred_values = pred_values + y_pred
#         except NameError:
#             pred_values = y_pred

#         # Voting.
#         y_pred_labels_1d = metric.get_predict_labels(y_pred)
#         pred_labels.append(y_pred_labels_1d)
#     pred_labels = np.array(pred_labels)

#     print('\n== Voting ==')
#     ensemble_y_pred = []
#     for i in range(pred_labels.shape[1]):
#         ensemble_y_pred.append(np.argmax(np.bincount(pred_labels[:, i])))

#     acc, f1, recall = metric.get(ensemble_y_pred, data['y_test'])
#     print('Accuracy: ' + str(acc))
#     print('F1 score: ' + str(f1))
#     print('Recall: ' + str(recall))

#     print('\n== Predictions sum ==')
#     ensemble_y_pred = np.argmax(pred_values, axis=1)

#     acc, f1, recall = metric.get(ensemble_y_pred, data['y_test'])
#     print('Accuracy: ' + str(acc))
#     print('F1 score: ' + str(f1))
#     print('Recall: ' + str(recall))
