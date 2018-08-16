from __future__ import division

import os
import sys
import time
import csv
from itertools import combinations
from datetime import datetime
from functools import reduce

# I don't want the Using xxx garbage.
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import keras.callbacks as cbks
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Input, concatenate, multiply
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.engine.topology import Layer

import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
import pandas as pd
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

tf.logging.set_verbosity(tf.logging.ERROR)
#np.random.seed(137)
#set_random_seed(2)

#########################################################################################

#run_name_prefix = 'regularized 2'
params = {
    'activation': 'tanh',
    'verbose': 0,
    'batch_size': 500000,
    'epochs': 300,
    'repetitions': 1,
    'fc_hidden_u': 20,
    'dropout': 0.2,
}

# For hardcoded context layers
params['cols'] = {
    'activity': [0, 1, 2, 5, 7, 8, 9, 28, 30, 32, 34, 36, 39],
    'peers': [3, 6, 10, 29, 31, 33, 35, 37, 40],
    'courseinfo': [4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 38],
}
params['cols']['ctx'] = params['cols']['peers'] + params['cols']['courseinfo']

params['merge_function'] = lambda layers: concatenate(layers, axis=1)

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

if len(sys.argv) < 3:
    print('Error: We need, at least, one training dataset and a testing dataset.');
    exit(1)

# The first argument is this file.
sys.argv.pop(0)
test_dataset = sys.argv.pop()
try:
    run_name_prefix
except:
    run_name_prefix = test_dataset

# Loading in memory. Not a massive issue as these datasets are relatively small.
test_file="datasets/" + test_dataset + ".csv"
train_files = []
for train_dataset_name in sys.argv:
    train_files.append("datasets/" + train_dataset_name + ".csv"),

data = {
}
data['x_train'], data['y_train'] = get_training_samples(train_files)
data['x_test'], data['y_test'] = get_testing_samples(test_file)
data['n_classes'] = data['y_train'].shape[1]
data['n_features'] = data['x_train'].shape[1]

print('Testing ' + test_dataset + ' with ' + str(len(train_files)) + ' training datasets')
print('  Data train size: '+ str(data['x_train'].shape[0]))
print('  Data test size: ' + str(data['x_test'].shape[0]))
print('  Total num features: ' + str(data['n_features']))

class Metrics(cbks.Callback):
    def __init__(self, x_test, y_test, summaries):
        self.x_test = x_test
        self.y_test = y_test
        self.summaries = summaries
        self.x_mismatches = []

        self.scores = {
            'acc': 0.,
            'f1': 0.,
            'auc': 0.,
        }

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):

        y_pred = self.model.predict(self.x_test)
        y_pred_labels = y_pred.round()

        y_test_1d = np.argmax(self.y_test, axis=1)
        y_pred_labels_1d = np.argmax(y_pred_labels, axis=1)

        mismatches = np.array([y_test_1d != y_pred_labels_1d])
        x_mismatches = self.x_test.iloc[mismatches.flatten()]

        # Write into a file.
        with open('mismatches.csv', 'wb') as mismatches_file:
            wr = csv.writer(mismatches_file, quoting=csv.QUOTE_NONNUMERIC)
            for index, features in enumerate(x_mismatches.values.tolist()):
                wr.writerow(features + [str(y_test_1d[index])])

        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x_test)
        y_pred_labels = y_pred.round()

        y_test_1d = np.argmax(self.y_test, axis=1)
        y_pred_labels_1d = np.argmax(y_pred_labels, axis=1)

        self.scores['acc'] = accuracy_score(y_test_1d, y_pred_labels_1d)
        if self.summaries:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.scores['acc']
            summary_value.tag = 'val_epoch_accuracy'
            self.summaries.writer.add_summary(summary, epoch)

        self.scores['auc'] = round(roc_auc_score(self.y_test, y_pred), 4)
        if self.summaries:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.scores['auc']
            summary_value.tag = 'val_epoch_auc'
            self.summaries.writer.add_summary(summary, epoch)

        self.scores['f1'] = f1_score(y_test_1d, y_pred_labels_1d)
        if self.summaries:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.scores['f1']
            summary_value.tag = 'val_epoch_f1'
            self.summaries.writer.add_summary(summary, epoch)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def subselect_cols(x, cols):
    # Return the selected columns from the set of input features.
    return tf.gather(x, cols, axis=1)

def get_regularization(reg):

    if reg == False:
        kernel_reg = None
        activity_reg = None
    else:
        kernel_reg = regularizers.l1(0.01)
        activity_reg = None

    return kernel_reg, activity_reg

def run_iteration(index, model, params, data, name=''):

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    if params['verbose'] == 1 and index == 0:
        # Print summaries.
        model.summary()

    callbacks = []
    summary_name = name + '-keras-' + str(time.time()) + '-' + params['name']
    summaries = TensorBoard(log_dir='./summaries/' + summary_name, histogram_freq=0,
        batch_size=params['batch_size'], write_grads=True)
    callbacks.append(summaries)

    metrics = Metrics(data['x_test'], data['y_test'], summaries)
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

def CombinationsLayer(inputs, params, reg=False):
    """Input features C(n, r) combinations."""

    kernel_reg, activity_reg = get_regularization(reg)

    # Generate comb(n_features, r) combinations.
    combs = combinations(range(params['n_features']), 2)

    layers = []

    for i, combination in enumerate(combs):

        # This combination features as input.
        cols = K.constant(combination, dtype='int32')
        layer_name = 'combination-' + str(i)
        combination_input = Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

        # 1 single output unit.
        layers.append(Dense(1, activation=params['activation'],
                            name='W-' + layer_name, kernel_regularizer=kernel_reg,
                            activity_regularizer=activity_reg)(combination_input))

    return params['merge_function'](layers)

def NoContextLayer(inputs, params):
    """Filter out all context features (course info and course peers)."""

    cols = K.constant(params['cols']['activity'], dtype='int32')
    layer_name = 'student-activity'

    return Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

def NoCourseInfoLayer(inputs, params):
    """Filter out all context features."""

    col_indexes = (params['cols']['activity'] + params['cols']['peers'])

    cols = K.constant(col_indexes, dtype='int32')
    layer_name = 'student-activity-and-peers'

    return Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

def ContextualiseActivityLayer(inputs, params, reg=False):
    """Each no-context feature combined with a hardcoded list of context features."""

    kernel_reg, activity_reg = get_regularization(reg)

    layers = []
    for col_index in params['cols']['activity']:

        # This no-context feature + all context features as input.
        cols = K.constant([col_index] + params['cols']['ctx'], dtype='int32')
        layer_name = 'in-context-' + str(col_index)
        contextualised_input = Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

        # 'multiplier' units as num outputs.
        layers.append(Dense(1, activation=params['activation'], name='W-' + layer_name,
                      kernel_regularizer=kernel_reg, activity_regularizer=activity_reg)(contextualised_input))

    return params['merge_function'](layers)

def ContextualiseActivityAndOriginalActivity(inputs, params, reg=False):
    """Each no-context feature combined with a hardcoded list of context features + inputs.
    """

    kernel_reg, activity_reg = get_regularization(reg)

    # Inputs as they are.
    layers = [inputs]

    for col_index in params['cols']['activity']:

        # This no-context feature + all context features as input.
        cols = K.constant([col_index] + params['cols']['ctx'], dtype='int32')
        layer_name = 'in-context-' + str(col_index)
        contextualised_input = Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

        # 'multiplier' units as num outputs.
        layers.append(Dense(1, activation=params['activation'], name='W-' + layer_name,
                      kernel_regularizer=kernel_reg, activity_regularizer=activity_reg)(contextualised_input))

    # Add all original no-context inputs.
    activity_cols = K.constant(params['cols']['activity'], dtype='int32')
    layer_name = 'no-context'
    layers.append(Lambda(subselect_cols, arguments={'cols': activity_cols}, name=layer_name)(inputs))

    return params['merge_function'](layers)

def SplitActivityAndContextLayer(inputs, params, n_ctx_units=False, reg=False):
    """Split input features in context and no-context."""

    layers = []

    # Context features learn separately.
    cols = K.constant(params['cols']['ctx'], dtype='int32')
    layer_name = 'context'
    ctx_input = Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

    if n_ctx_units == False:
        # By default the number of units equal to the number of context cols.
        n_ctx_units = len(params['cols']['ctx'])

    kernel_reg, activity_reg = get_regularization(reg)

    ctx_layer = Dense(n_ctx_units, activation=params['activation'], name='W-' + layer_name,
                      kernel_regularizer=kernel_reg, activity_regularizer=activity_reg)(ctx_input)
    layers.append(ctx_layer)

    # Student activity context features learn separately.
    cols = K.constant(params['cols']['activity'], dtype='int32')
    layer_name = 'no-context'
    no_ctx_input = Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

    # Number of units equal to the number of no context features.
    layers.append(Dense(len(params['cols']['activity']), activation=params['activation'], name='W-' + layer_name)(no_ctx_input))

    return params['merge_function'](layers)

def SplitAllInputsLayer(inputs, params):
    """Multiple inputs separated based on a hardcoded set of cols."""

    layers = []
    for layer_name, layer_data in params['separate_cols'].items():

        cols = K.constant(layer_data['cols'], dtype='int32')
        layer_input = Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

        # Number of units equal to the number of columns.
        layers.append(Dense(layer_data['units'], activation=layer_data['activation'], kernel_regularizer=layer_data['kernelreg'],
                            activity_regularizer=layer_data['kernelreg'], name='W-' + layer_name)(layer_input))

    return params['merge_function'](layers)


def add_dropout(layer, params, name='dropout'):
    return Dropout(params['dropout'], name=name)(layer)

def add_fc(layer, params, n_units=False, name='main-hidden'):
    if n_units == False:
        n_units = params['fc_hidden_u']

    kernel_reg, activity_reg = get_regularization(False)

    return Dense(n_units, activation=params['activation'], name=name,
                 kernel_regularizer=kernel_reg, activity_regularizer=activity_reg)(layer)

def add_softmax(layer, params, name='main-output'):
    return Dense(params['n_classes'], activation='softmax', name=name)(layer)

def baseline(params):

    input_layer = Input(shape=(params['n_features'],), name='inputs')

    if params['dataset'] == 'nocontext':
        # Filter out context (course info + peers columns).
        base_layer = NoContextLayer(input_layer, params)
    elif params['dataset'] == 'withpeers':
        # Filter out course info.
        base_layer = NoCourseInfoLayer(input_layer, params)
    elif params['dataset'] == 'all':
        base_layer = input_layer

    return input_layer, base_layer

# 1 hidden and fully connected nn.
def nn_1h(params):
    input_layer, base_layer = baseline(params)
    base_layer = add_fc(base_layer, params)
    base_layer = add_dropout(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

# Extra hidden layer.
def nn_2h(params):
    input_layer, base_layer = baseline(params)
    base_layer = add_fc(base_layer, params, name='pre-hidden')
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

def nn_2h_reg(params):
    input_layer, base_layer = baseline(params)
    base_layer = add_fc(base_layer, params, name='extra-hidden')
    base_layer = add_dropout(base_layer, params)
    base_layer = Dense(params['fc_hidden_u'], activation=params['activation'],
                       kernel_regularizer=regularizers.l2(0.01),
                       activity_regularizer=regularizers.l1(0.01),
                       name='pre-hidden')(base_layer)
    output = add_softmax(base_layer, params)
    return input_layer, output

# Combinations of 'r' features.
def comb(params):
    input_layer, base_layer = baseline(params)
    base_layer = CombinationsLayer(base_layer, params)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

def comb_reg(params):
    input_layer, base_layer = baseline(params)
    base_layer = CombinationsLayer(base_layer, params, reg=True)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

# In-context features.
def inctx(params):
    input_layer, base_layer = baseline(params)
    base_layer = ContextualiseActivityLayer(base_layer, params)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

# In-context features.
def inctx_reg(params):
    input_layer, base_layer = baseline(params)
    base_layer = ContextualiseActivityLayer(base_layer, params, reg=True)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

# In-context features + original inputs.
def inctx_extra(params):
    input_layer, base_layer = baseline(params)
    base_layer = ContextualiseActivityAndOriginalActivity(base_layer, params)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

# In-context features + original inputs.
def inctx_extra_reg(params):
    input_layer, base_layer = baseline(params)
    base_layer = ContextualiseActivityAndOriginalActivity(base_layer, params, reg=True)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

# Simple context / no-context separation.
def simple_separate_all(params):
    input_layer, base_layer = baseline(params)
    base_layer = SplitActivityAndContextLayer(base_layer, params)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

def simple_separate_all_reg(params):
    input_layer, base_layer = baseline(params)
    base_layer = SplitActivityAndContextLayer(base_layer, params, reg=True)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

def simple_separate_1(params):
    input_layer, base_layer = baseline(params)
    base_layer = SplitActivityAndContextLayer(base_layer, params, n_ctx_units=1)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

def simple_separate_1_reg(params):
    input_layer, base_layer = baseline(params)
    base_layer = SplitActivityAndContextLayer(base_layer, params, n_ctx_units=1, reg=True)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

# Separate inputs, single output.
def complex_separate(params):
    input_layer, base_layer = baseline(params)
    base_layer = SplitAllInputsLayer(base_layer, params)
    base_layer = add_dropout(base_layer, params)
    base_layer = add_fc(base_layer, params)
    output = add_softmax(base_layer, params)
    return input_layer, output

models = []
#models.append({
    #'name': 'All features NN - 1 hidden.',
    #'build': nn_1h,
    #'dataset': 'all'
#})

#models.append({
    #'name': 'No context NN - 1 hidden.',
    #'build': nn_1h,
    #'dataset': 'nocontext'
#})

#models.append({
    #'name': 'With peers NN - 1 hidden.',
    #'build': nn_1h,
    #'dataset': 'withpeers'
#})

#models.append({
    #'name': 'All features NN - 2 hidden.',
    #'build': nn_2h,
    #'dataset': 'all'
#})

#models.append({
    #'name': 'With peers NN - 2 hidden.',
    #'build': fc_2h,
    #'dataset': 'withpeers'
#})

#models.append({
    #'name': 'No context NN - 2 hidden.',
    #'build': nn_2h,
    #'dataset': 'nocontext'
#})

#models.append({
    #'name': 'Activity features trained with context.',
    #'build': inctx,
    #'dataset': 'all'
#})

#models.append({
    #'name': 'Activity features trained with context (REG).',
    #'build': inctx_reg,
    #'dataset': 'all'
#})

models.append({
    'name': 'Activity features trained with context + original activity features.',
    'build': inctx_extra,
    'dataset': 'all'
})

models.append({
    'name': 'Activity features trained with context + original activity features (REG).',
    'build': inctx_extra_reg,
    'dataset': 'all'
})

#models.append({
    #'name': 'Separate no-context / 1 context inputs.',
    #'build': simple_separate_1,
    #'dataset': 'all'
#})

#models.append({
    #'name': 'Separate no-context / 1 context inputs (REG).',
    #'build': simple_separate_1_reg,
    #'dataset': 'all'
#})

#models.append({
    #'name': 'Separate no-context / all context inputs.',
    #'build': simple_separate_all,
    #'dataset': 'all'
#})

#models.append({
    #'name': 'Separate no-context / all context inputs (REG).',
    #'build': simple_separate_all_reg,
    #'dataset': 'all'
#})

#models.append({
    #'name': 'Separate activity / course peers / required.',
    #'build': complex_separate,
    #'dataset': 'all'
#})

#models.append({
    #'name': 'Combinations of features in pairs.',
    #'build': comb,
    #'dataset': 'all'
#})

#models.append({
    #'name': 'Combinations of features in pairs (REG).',
    #'build': comb_reg,
    #'dataset': 'all'
#})

model_scores = []
for model_data in models:

    start_time = datetime.now()

    print('\nModel ' + model_data['name'])

    acc = []
    f1 = []
    auc = []
    for index in range(params['repetitions']):

        params['n_features'] = data['n_features']
        params['n_classes'] = data['n_classes']
        params['name'] = model_data['name']
        params['dataset'] = model_data['dataset']

        inputs, output = model_data['build'](params)

        model = Model(inputs, output)

        if index == 0:
            print('  Total params: ' + str(model.count_params()))
        score = run_iteration(index, model, params, data, name=run_name_prefix)
        acc.append(score['acc'])
        f1.append(score['f1'])
        auc.append(score['auc'])

    model_score = {
        'name': model_data['name'],
        'acc': np.average(acc),
        'f1': np.average(f1),
        'auc': np.average(auc),
    }
    model_scores.append(model_score)

    print('  Time: ' + str(datetime.now() - start_time))
    print('  Median accuracy: ' + str(np.median(acc)))
    print('  Average accuracy: ' + str(model_score['acc']))
    print('  Median AUC: ' + str(np.median(auc)))
    print('  Average AUC: ' + str(model_score['auc']))
    print('  Median F1: ' + str(np.median(f1)))
    print(' *Average F1: ' + str(model_score['f1']))

print('\nResults for ' + test_dataset + ' ready for copy & paste:\n')
for model_score in model_scores:
    print(model_score['name'] + ', ' + str(model_score['f1']) + ', ' + str(model_score['acc']) + ', ' + str(model_score['auc']))
