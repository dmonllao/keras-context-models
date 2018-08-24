import time

import numpy as np
#np.random.seed(137)

# Local packages.
import layer
import dataset
import debugger

def test(network, datasets, params, run_prefix, model_scores):

    # Local import as this is using multiprocessing.
    from keras.models import Model
    import keras.backend as K
    import tensorflow as tf

    K.set_session(tf.Session())

    data = datasets[network['dataset']]

    start_time = time.time()
    print('\nModel ' + network['name'] + ' using ' + network['dataset'] + ' as test dataset')

    acc = []
    f1 = []
    recall = []
    for index in range(params['repetitions']):

        params['n_features'] = data['n_features']
        params['n_classes'] = data['n_classes']
        params['name'] = network['name']
        params['feature_set'] = network['feature_set']

        if 'options' in network.keys():
            options = network['options']
        else:
            options = {}

        # Build the network
        inputs, output = network['network'](params, options)

        model = Model(inputs, output)

        if params['verbose'] == 1 and index == 0:
            print('  Total params: ' + str(model.count_params()))

        score = test_model(index, model, params, data, name=run_prefix)
        acc.append(score['acc'])
        f1.append(score['f1'])
        recall.append(score['recall'])

        # Clear current session for the next run to avoid histogram problems
        # https://github.com/keras-team/keras/issues/4499#issuecomment-279723967
        K.clear_session()

    print('  Time: ' + str(time.time() - start_time))
    #print('  Median accuracy: ' + str(np.median(acc)))
    #print('  Average accuracy: ' + str(model_score['acc']))
    #print('  Median F1: ' + str(np.median(f1)))
    #print(' *Average F1: ' + str(model_score['f1']))
    #print('  Median recall: ' + str(np.median(recall)))
    #print(' *Average recall: ' + str(model_score['recall']))

    model_score = {
        'name': network['name'],
        'dataset': network['dataset'],
        'acc': np.average(acc),
        'f1': np.average(f1),
        'recall': np.average(recall),
    }
    model_scores[network['dataset'] + '-' + network['name']] = model_score

def test_model(index, model, params, data, name=None):

    from keras.optimizers import Adam
    from keras.callbacks import TensorBoard

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=params['lr']),
                  metrics=['accuracy'])

    if params['verbose'] == 1 and index == 0:
        # Print summaries.
        model.summary()

    summary_name = data['test_dataset_id'] + '-' + str(time.time()) + '-' + params['name']
    if name != None:
        summary_name = name + '-' + summary_name

    # Histograms need a validation_split.
    if params['debug'] == 1:
        histogram_freq = 30
        validation_split = 0.05
    else:
        histogram_freq = 0
        validation_split = 0.

    callbacks = []
    summaries = TensorBoard(log_dir='./summaries/' + summary_name, histogram_freq=histogram_freq,
        batch_size=params['batch_size'], write_grads=True)
    callbacks.append(summaries)

    metrics = debugger.Metrics(data['x_test'], data['y_test'], summaries, data['test_dataset_id'])
    callbacks.append(metrics)

    model.fit(data['x_train'], data['y_train'],
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                verbose=params['verbose'],
                validation_split=validation_split,
                callbacks=callbacks,
                shuffle=True)

    print(metrics.scores)
    return metrics.scores

def get_combinations(args, models):

    dataset_list = dataset.test_dataset_list(args.test_datasets)

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
    for dataset_id in dataset_list:
        for model_data in test_models:
            network = model_data.copy()
            network['dataset'] = dataset_id
            networks.append(network)

    return networks

def fc(params, options):
    input_layer, base_layer = layer.baseline(params)

    if options['n_layers'] == 2:
        base_layer = layer.add_fc(base_layer, params, name='pre-hidden')
        base_layer = layer.add_dropout(base_layer, params)

    base_layer = layer.add_fc(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# In-context features.
def inctx(params, options):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.ContextualiseActivity(base_layer, params, reg=options['reg'])
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# In-context features + original inputs.
def inctx_extra(params, options):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.ContextualiseActivityAndOriginalActivity(base_layer, params, reg=options['reg'])
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# Simple context / no-context separation.
def simple_separate(params, options):

    if 'n_ctx_units' in options:
        n_ctx_units = options['n_ctx_units']
    else:
        # Will default to the number of context features.
        n_ctx_units = False

    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.SplitActivityAndContext(base_layer, params, n_ctx_units=n_ctx_units,
                                               reg=options['reg'])
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# Separate inputs, single output.
def complex_separate(params):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.SplitAllInputs(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output

# Combinations of 'r' features.
def comb(params, options):
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.Combinations(base_layer, params, options['reg'])
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output
