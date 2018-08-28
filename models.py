import network

def get_all():

    models = []

    models.append({
        'name': 'No context NN - 1 hidden.',
        'network': network.fc,
        'feature_set': 'nocontext',
        'options': {'n_layers': 1},
    })

    models.append({
        'name': 'With peers NN - 2 hidden.',
        'network': network.fc,
        'feature_set': 'withpeers',
        'options': {'n_layers': 2},
    })

    models.append({
        'name': 'All features NN - 2 hidden.',
        'network': network.fc,
        'feature_set': 'all',
        'options': {'n_layers': 2},
    })

    models.append({
        'name': 'Activity features trained with context.',
        'network': network.inctx,
        'feature_set': 'all',
        'options': {'reg': False},
    })

    models.append({
        'name': 'Activity features trained with context + original activity features.',
        'network': network.inctx_extra,
        'feature_set': 'all',
        'options': {'reg': False},
    })

    models.append({
        'name': 'Separate no-context / 1 context inputs.',
        'network': network.simple_separate,
        'feature_set': 'all',
        'options': {
            'reg': True,
            'n_ctx_units': 1,
            'context_includes_peers': True,
        },
    })

    models.append({
        'name': 'Separate activity & peers / course inputs.',
        'network': network.simple_separate,
        'feature_set': 'all',
        'options': {
            'reg': True,
            'n_ctx_units': 1,
            'context_includes_peers': False,
        },
    })

    models.append({
        'name': 'Separate no-context / all context inputs.',
        'network': network.simple_separate,
        'feature_set': 'all',
        'options': {
            'reg': True,
            'context_includes_peers': True,
        },
    })

    models.append({
        'name': 'Separate activity / course / peers.',
        'network': network.complex_separate,
        'feature_set': 'all',
    })

    #models.append({
        #'name': 'Combinations of features in pairs.',
        #'network': network.comb,
        #'feature_set': 'all',
        #'options': {'reg': False},
    #})

    return models
