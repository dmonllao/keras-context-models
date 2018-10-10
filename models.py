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
        'name': 'All features NN - 2 hidden.',
        'network': network.fc,
        'feature_set': 'all',
        'options': {'n_layers': 2},
    })

    return models
