from __future__ import division

import os
import sys
import time
from itertools import combinations
from datetime import datetime

from keras import regularizers
from keras.layers import Dense, Dropout, Lambda, Input, concatenate, multiply
from keras import backend as K
from keras.engine.topology import Layer

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

def add_dropout(layer, params, name='dropout'):
    return Dropout(params['dropout'], name=name)(layer)

def add_fc(layer, params, n_units=False, regularization=True, name='main-hidden'):
    if n_units == False:
        n_units = params['fc_hidden_u']

    kernel_reg, activity_reg = get_regularization(regularization)

    return Dense(n_units, activation=params['activation'], name=name,
                 kernel_regularizer=kernel_reg, activity_regularizer=activity_reg)(layer)

def add_softmax(layer, params, name='main-output'):
    return Dense(params['n_classes'], activation='softmax', name=name)(layer)

def baseline(params):

    input_layer = Input(shape=(params['n_features'],), name='inputs')

    if params['feature_set'] == 'nocontext':
        # Filter out context (course info + peers columns).
        base_layer = NoContext(input_layer, params)
    elif params['feature_set'] == 'withpeers':
        # Filter out course info.
        base_layer = NoCourseInfo(input_layer, params)
    elif params['feature_set'] == 'all':
        base_layer = input_layer

    return input_layer, base_layer

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

def Combinations(inputs, params, reg=False):
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

    return concatenate(layers, axis=1)

def NoContext(inputs, params):
    """Filter out all context features (course info and course peers)."""

    cols = K.constant(params['cols']['activity'], dtype='int32')
    layer_name = 'student-activity'

    return Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

def NoCourseInfo(inputs, params):
    """Filter out all context features."""

    col_indexes = (params['cols']['activity'] + params['cols']['peers'])

    cols = K.constant(col_indexes, dtype='int32')
    layer_name = 'student-activity-and-peers'

    return Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

def ContextualiseActivity(inputs, params, reg=False):
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

    return concatenate(layers, axis=1)

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

    return concatenate(layers, axis=1)

def SplitActivityAndContext(inputs, params, n_ctx_units=False, reg=False):
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

    return concatenate(layers, axis=1)

def SplitAllInputs(inputs, params):
    """Multiple inputs separated based on a hardcoded set of cols."""

    layers = []
    for layer_name, layer_data in params['separate_cols'].items():

        cols = K.constant(layer_data['cols'], dtype='int32')
        layer_input = Lambda(subselect_cols, arguments={'cols': cols}, name=layer_name)(inputs)

        # Number of units equal to the number of columns.
        layers.append(Dense(layer_data['units'], activation=layer_data['activation'], kernel_regularizer=layer_data['kernelreg'],
                            activity_regularizer=layer_data['kernelreg'], name='W-' + layer_name)(layer_input))

    return concatenate(layers, axis=1)


