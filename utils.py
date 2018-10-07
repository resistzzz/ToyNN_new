"""

"""

import numpy as np


def one_hot(labels, num):

    # labels: N (0-num-1)
    new_labels = np.zeros((labels.shape[0], num))
    new_labels[range(labels.shape[0]), labels] = 1

    return new_labels

def testResult2labels(y_test):

    # y_test: N*num
    labels = np.argmax(y_test, axis=1)

    return labels
