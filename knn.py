from operator import itemgetter

import numpy as np


class DataSet:
    """
    A class that contains input features and output labels.
    """

    def __init__(self, input, labels):
        """

        :param input: m X n array, m is the num of samples, n is the num of features.
        :param labels: A list contains all samples' label.
        :return: DataSet
        """
        self.input = input
        self.labels = labels


def clustered(k, target, samples):
    """
    :param k: K-Nearest Neighbor's k
    :param target: the target that you want to cluster, numpy.array
    :param samples: all the training samples, it must be DataSet
    :return: the cluster.
    """
    sample_num = samples.input.shape[0]
    helpers = np.tile(target, (sample_num, 1))
    distance = _distance(helpers, samples.input)
    sorted_indices = distance.argsort()
    class_cluster = {}
    for i in range(k):
        index = sorted_indices[i]
        one_class = samples.labels[index]
        if one_class not in class_cluster.keys():
            class_cluster[one_class] = 1
        else:
            class_cluster[one_class] += 1
    sorted_cluster = sorted(class_cluster.items(), key=itemgetter(1), reverse=True)
    return sorted_cluster[0][0]


def _distance(arr1, arr2):
    """
    :type arr1: ndarray or matrix
    :type arr2: ndarray or matrix
    :return: array
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Two array must be same dimension")
    diff = np.power(arr1 - arr2, 2)
    diff_sum = np.sum(diff, axis=1)
    return np.sqrt(diff_sum)
