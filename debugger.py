import csv
import os
import time

import keras.callbacks as cbks
import numpy as np

import metric

class Metrics(cbks.Callback):
    def __init__(self, x_test, y_test, summaries, test_dataset_id):
        self.x_test = x_test
        self.y_test = y_test
        self.summaries = summaries
        self.test_dataset_id = test_dataset_id
        self.x_mismatches = []

        self.scores = {
            'acc': 0.,
            'f1': 0.,
            'recall': 0.,
        }

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):

        # y_pred = self.model.predict(self.x_test)
        # y_pred_labels = y_pred.round()

        # y_test_1d = np.argmax(self.y_test, axis=1)
        # y_pred_labels_1d = np.argmax(y_pred_labels, axis=1)

        # mismatches = np.array([y_test_1d != y_pred_labels_1d])
        # x_mismatches = self.x_test[mismatches.flatten()]

        # # Write into a file.
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # filename = self.test_dataset_id + '-' + str(int(time.time())) + '.csv'

        # mismatches_filepath = os.path.join(
        #     script_dir, 'mismatches', filename)
        # with open(mismatches_filepath, 'wb') as mismatches_file:
        #     wr = csv.writer(mismatches_file, quoting=csv.QUOTE_NONNUMERIC)
        #     for index, features in enumerate(x_mismatches):
        #         wr.writerow(features + [y_test_1d[index]])

        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        # For multiprocessing.
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)

        y_pred_labels_1d = metric.get_predict_labels(
            self.model.predict(self.x_test)
        )

        self.scores['acc'], self.scores['f1'], self.scores['recall'] = \
            metric.get(y_pred_labels_1d, self.y_test)

        if self.summaries:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.scores['acc']
            summary_value.tag = 'val_epoch_accuracy'
            self.summaries.writer.add_summary(summary, epoch)

        if self.summaries:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.scores['f1']
            summary_value.tag = 'val_epoch_f1'
            self.summaries.writer.add_summary(summary, epoch)

        if self.summaries:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.scores['recall']
            summary_value.tag = 'val_epoch_recall'
            self.summaries.writer.add_summary(summary, epoch)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
