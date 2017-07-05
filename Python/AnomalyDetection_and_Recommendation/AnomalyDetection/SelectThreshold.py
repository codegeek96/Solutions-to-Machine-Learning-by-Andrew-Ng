"""Find the best threshold (epsilon) to use for selecting outliers
"""

import numpy as np


def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        predictions = pval < epsilon

        tp = np.sum(np.logical_and(predictions == 1, yval == 1))
        fp = np.sum(np.logical_and(predictions == 1, yval == 0))
        fn = np.sum(np.logical_and(predictions == 0, yval == 1))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1
