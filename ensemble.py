import numpy as np

import debugger


def vote(y_preds_by_dataset):

    all_preds = []
    for dataset_id, data in y_preds_by_dataset.items():
        metrics = debugger.Metrics(
            data['y_test'], False, False, False)

        y_pred = []
        n_samples = data['y_pred'].shape[1]
        for i in range(n_samples):
            max_voted = np.argmax(np.bincount(data['y_pred'][:, i]))
            y_pred.append(max_voted)

        y_pred = np.array(y_pred)
        scores = metrics.get_pred_metrics(y_pred)
        print('Votes:')
        print('    Acc: ' + str(scores[0]))
        print('    F1: ' + str(scores[1]))
        print('    Recall: ' + str(scores[2]))
