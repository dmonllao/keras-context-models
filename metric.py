import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def get(y_pred_labels_1d, y_test):
    y_test_1d = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_test_1d, y_pred_labels_1d)
    f1 = f1_score(y_test_1d, y_pred_labels_1d)
    recall = recall_score(y_test_1d, y_pred_labels_1d)

    return acc, f1, recall

def get_predict_labels(y_pred):
    y_pred_labels = y_pred.round()
    return np.argmax(y_pred_labels, axis=1)
