import network

def get_all():

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

    return models
