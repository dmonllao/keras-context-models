import time

import numpy as np
# np.random.seed(137)

# Local packages.
import layer
import dataset
import debugger


def test(network, datasets, params, run_prefix):

    # Local import as this is using multiprocessing.
    from keras.models import Model
    import keras.backend as K
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)

    data = datasets[network['dataset']]

    K.set_session(tf.Session())

    start_time = time.time()
    print('\nModel ' + network['name'] + ' using ' +
          network['dataset'] + ' as test dataset')

    params, options = parse_network_obj(network, params,
                                        data['n_features'],
                                        data['n_classes'])
    acc = []
    f1 = []
    recall = []
    for index in range(params['repetitions']):

        # Build the network
        inputs, output = network['network'](params, options)
        model = Model(inputs, output)
        model = compile_model(model, params)

        score = test_model(index, model, params, data, name=run_prefix)
        acc.append(score['acc'])
        f1.append(score['f1'])
        recall.append(score['recall'])

        # Clear current session for the next run to avoid histogram problems
        # https://github.com/keras-team/keras/issues/4499#issuecomment-279723967
        # COMMENTED OUT JUST TO PREDICT LATER.
        # K.clear_session()

    # print('  Time: ' + str(time.time() - start_time))

    # Attaching the last repetition Model object.
    return {
        'model': model,
        'name': network['name'],
        'dataset': network['dataset'],
        'acc': np.average(acc),
        'f1': np.average(f1),
        'recall': np.average(recall),
    }


def compile_model(model, params):

    from keras.optimizers import Adam

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=params['lr']),
                  metrics=['accuracy'])
    return model


def test_model(index, model, params, data, name=None):

    callbacks = get_fit_callbacks(
        params, data, name)
    kwargs = get_fit_kwargs(params, callbacks)

    if params['verbose'] == 1 and index == 0:
        print('  Total params: ' + str(model.count_params()))
        # Print summaries.
        model.summary()

    model.fit(data['x_train'], data['y_train'], **kwargs)

    # callbacks[1] is a Metrics instance.
    # print(callbacks[1].scores)
    return callbacks[1].scores


def parse_network_obj(network, params, n_features, n_classes):

    params['n_features'] = n_features
    params['n_classes'] = n_classes
    params['name'] = network['name']
    params['feature_set'] = network['feature_set']

    if 'options' in network.keys():
        options = network['options']
    else:
        options = {}

    return params, options


def get_fit_callbacks(params, data, name):
    from keras.callbacks import TensorBoard

    summary_name = data['test_dataset_id'] + '-' + params['name'] + '-' + \
        str(time.time())
    if name is not None:
        summary_name = name + '-' + summary_name

    # Histograms need a validation_split.
    if params['debug'] == 1:
        histogram_freq = 30
    else:
        histogram_freq = 0

    callbacks = []
    summaries = TensorBoard(log_dir='./summaries/' + summary_name,
                            histogram_freq=histogram_freq,
                            batch_size=params['batch_size'], write_grads=True)
    callbacks.append(summaries)

    metrics = debugger.Metrics(
        data['x_test'], data['y_test'], summaries, data['test_dataset_id'])
    callbacks.append(metrics)

    return callbacks


def get_fit_kwargs(params, callbacks=None):

    if params['debug'] == 1:
        validation_split = 0.05
    else:
        validation_split = 0.

    kwargs = {
        'batch_size': params['batch_size'],
        'epochs': params['epochs'],
        'verbose': params['verbose'],
        'validation_split': validation_split,
        'shuffle': True
    }

    if callbacks is not None:
        kwargs['callbacks'] = callbacks

    return kwargs


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


def get_separate_training_datasets_combinations(args, models):

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

    networks = {}
    for dataset_id in dataset_list:

        for model_data in test_models:
            key = dataset_id + '-' + model_data['name']
            networks[key] = []

            for train_dataset_id in dataset.it_them():
                if train_dataset_id == dataset_id:
                    continue

                network = model_data.copy()
                network['dataset'] = dataset_id + '-' + train_dataset_id
                networks[key].append(network)

    return networks


def fc(params, options):
    input_layer, base_layer = layer.baseline(params)

    if options['n_layers'] == 2:
        base_layer = layer.add_fc(
            base_layer, params, name='pre-hidden', regularization=True)
        base_layer = layer.add_dropout(base_layer, params, name='pre-dropout')

    base_layer = layer.add_fc(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output


def inctx(params, options):
    """In-context features."""
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.ContextualiseActivity(
        base_layer, params, reg=options['reg'])
    base_layer = layer.add_dropout(base_layer, params, name='base-dropout')
    base_layer = layer.add_fc(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output


def inctx_extra(params, options):
    """In-context features + original inputs."""

    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.ContextualiseActivityAndOriginalActivity(
        base_layer, params, reg=options['reg'])
    base_layer = layer.add_dropout(base_layer, params, name='base-dropout')
    base_layer = layer.add_fc(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output


def simple_separate(params, options):
    """Simple context / no-context separation."""

    if 'n_ctx_units' in options:
        n_ctx_units = options['n_ctx_units']
    else:
        # Will default to the number of context features.
        n_ctx_units = False

    peers = options['context_includes_peers']

    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.SplitActivityAndContext(base_layer, params,
                                               n_ctx_units=n_ctx_units,
                                               reg=options['reg'],
                                               context_includes_peers=peers)
    base_layer = layer.add_dropout(base_layer, params, name='base-dropout')
    base_layer = layer.add_fc(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output


def complex_separate(params, options):
    """Separate inputs, single output."""
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.SplitAllInputs(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params, name='base-dropout')
    base_layer = layer.add_fc(base_layer, params)
    base_layer = layer.add_dropout(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output


def comb(params, options):
    """Combinations of 'r' features."""
    input_layer, base_layer = layer.baseline(params)
    base_layer = layer.Combinations(base_layer, params, options['reg'])
    base_layer = layer.add_dropout(base_layer, params)
    base_layer = layer.add_fc(base_layer, params)
    output = layer.add_softmax(base_layer, params)
    return input_layer, output
