"""
Utility code for miscellaneous tasks

Created by Matthew Keaton on 4/16/2020
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()


def elapsed_time(seconds, short=False):
    if not short:
        minutes = int(seconds // 60)
        hours = int(minutes // 60)
        e_time = ''
        if hours > 0:
            if hours == 1:
                e_time += '1 hour, '
            else:
                e_time += '{:d} hours, '.format(hours)
            minutes %= 60
            seconds %= 60
        if minutes > 0:
            if minutes == 1:
                e_time += '1 minute, '
            else:
                e_time += '{:d} minutes, '.format(minutes)
            seconds %= 60
        if seconds == 1:
            e_time += '1 second.'
        else:
            e_time += '{:.1f} seconds.'.format(seconds)
        return e_time
    else:
        minutes = int(seconds // 60)
        hours = int(minutes // 60)
        e_time = ''
        if hours > 0:
            if hours == 1:
                e_time += '1hr '
            else:
                e_time += '{:d}hrs '.format(hours)
            minutes %= 60
            seconds %= 60
        if minutes > 0:
            if minutes == 1:
                e_time += '1m '
            else:
                e_time += '{:d}m '.format(minutes)
            seconds %= 60
        if seconds == 1:
            e_time += '1s'
        else:
            e_time += '{:.1f}s'.format(seconds)
        return e_time


def create_confusion_matrix(y_true, y_pred, classes, normalize=None):
    # Get data into confusion matrix (array)
    num_classes = len(classes)
    size = len(y_true)
    cols = ['True Values', 'Predicted Values', 'values']
    conf_mat = np.zeros((num_classes, num_classes))
    for i in range(size):
        conf_mat[y_true[i], y_pred[i]] += 1

    # Normalization procedure
    if normalize == 'all':
        conf_mat = np.divide(conf_mat, size)
    elif normalize == 'True':
        true_sum = np.sum(conf_mat, axis=1)
        cm_t = np.transpose(conf_mat)
        conf_mat = np.transpose(np.divide(cm_t, true_sum))
    elif normalize == 'Pred':
        pred_sum = np.sum(conf_mat, axis=0)
        conf_mat = np.divide(conf_mat, pred_sum)
    conf_frame = pd.DataFrame([], columns=cols)
    for i in range(num_classes):
        for j in range(num_classes):
            new_row = pd.DataFrame([[classes[i], classes[j], conf_mat[i][j]]], columns=cols)
            conf_frame = conf_frame.append(new_row)
    conf_pivot = conf_frame.pivot(index=cols[0], columns=cols[1], values=cols[2])
    conf_pivot = conf_pivot.reindex(classes)
    conf_pivot = conf_pivot.reindex(columns=classes)

    if normalize:
        return conf_pivot, sns.heatmap(conf_pivot, annot=True, cmap="BuGn", fmt='.3f', cbar=False)
    else:
        return conf_pivot, sns.heatmap(conf_pivot, annot=True, cmap="BuGn", fmt='g', cbar=False)
