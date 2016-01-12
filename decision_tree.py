from operator import itemgetter
import numpy as np
import pickle
from math import log


class _TreeNote:
    """
    This is a internal object that holds the current labeled feature and subtrees.

    """
    def __init__(self, value, index):
        self.sub_trees = {}
        self.value = value
        self.index = index

    def add_subtree(self, key, subtree):
        self.sub_trees[key] = subtree


class DecisionTree:
    """
    Decision Tree object handles basic operations
    """

    @staticmethod
    def load_tree(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def __init__(self):
        self.root = None

    @staticmethod
    def _calculate_shannon_entropy(output):
        """
        Calculate shannon entropy: H = - sum(p(xi) * log2p(xi))
        :param sample: ndarray
        :return: entropy of the current array
        """
        num_entries = len(output)
        num_classes = {}
        for current in output:
            if current not in num_classes.keys():
                num_classes[current] = 1
            else:
                num_classes[current] += 1
        entropy = 0.0
        for key in num_classes:
            p = float(num_classes[key] / num_entries)
            entropy -= p * log(p, 2)
        return entropy

    @staticmethod
    def _split_data_set(sample, output, axis, value, feature_label=[]):
        if axis < 0 or axis > sample.shape[1]:
            raise ValueError("ERROR: Axis " + str(axis) + " is out of bounds: " + str(sample.shape[1]))
        new_sample = []
        new_output = []
        for i in range(sample.shape[0]):
            feature_vec = sample[i, :]
            if feature_vec[axis] == value:
                reduced_feature = feature_vec[0:axis]
                reduced_feature = np.append(reduced_feature, feature_vec[(axis+1):])
                new_sample.append(reduced_feature)
                new_output.append(output[i])
        sub_feature_labels = []
        if len(feature_label) > 0:
            sub_feature_labels = feature_label[0:axis]
            sub_feature_labels.extend(feature_label[axis+1:])
        return np.array(new_sample), new_output, sub_feature_labels

    def _select_feature_to_split(self, sample, output):
        num_samples, num_features = sample.shape
        initial_entropy = self._calculate_shannon_entropy(output)
        best_info_gain = 0
        feature_axis = -1
        for i in range(num_features):
            unique_values = np.unique(sample[:, i])
            entropy = 0.0
            for value in unique_values:
                data_set, split_output, _ = self._split_data_set(sample, output, i, value)
                prob = float(data_set.shape[0] / num_samples)
                entropy += prob * self._calculate_shannon_entropy(split_output)
            info_gain = initial_entropy - entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                feature_axis = i
        return feature_axis

    @staticmethod
    def _vote_count(output):
        class_cluster = {}
        for value in output:
            if value not in class_cluster.keys():
                class_cluster[value] = 1
            else:
                class_cluster[value] += 1
        sorted_cluster = sorted(class_cluster.items(), key=itemgetter(1), reverse=True)
        return sorted_cluster[0][0]

    def train(self, sample, output, store_path='', feature_label=[]):
        if sample.size == 0:
            raise ValueError("ERROR: Cannot use empty sample")
        if len(output) == 0:
            raise ValueError("ERROR: Cannot use empty output")
        if sample.shape[0] != len(output):
            raise ValueError("Error: The number of training sample must equal to the number of output")
        self._train(sample, output, feature_label=feature_label, node=self.root)
        if len(store_path) != 0:
            with open(store_path, 'wb') as output_file:
                pickler = pickle.Pickler(output_file, -1)
                pickler.dump(self)

    def predict(self, target):
        if len(target) == 0:
            raise ValueError('ERROR: Cannot use empty input')
        if self.root is None:
            raise RuntimeError("ERROR: tree hasn't been trained yet, train the tree first.")
        return self._predict(target, self.root)

    def _predict(self, target, node: _TreeNote):
        if len(node.sub_trees) == 0:
            return node.value
        feature_to_check = node.index
        feature_value = target[feature_to_check]

        sub_tree = node.sub_trees[feature_value]
        if not sub_tree:
            return sub_tree.name
        else:
            return self._predict(target, sub_tree)

    def _train(self, sample, output, current_feature_value=-1, feature_label=[], node=None):
        """
        Recursively generate decision tree. This is private and should not be used. Use train method instead.
        :param sample: numpy ndarray
        :param output: list
        :param current_feature_value: int, default 0
        :param feature_label: list
        :param node: current node
        """
        unique_output = np.unique(output)
        if unique_output.size == 1:
            if self.root is None:
                self.root = _TreeNote(output[0], 0)
            else:
                node.add_subtree(current_feature_value, _TreeNote(output[0], 0))
            return
        if sample.shape[1] == 0:
            value = self._vote_count(output)
            if self.root is None:
                self.root = _TreeNote(value, 0)
            else:
                node.add_subtree(current_feature_value, _TreeNote(value, 0))
            return

        feature_to_split = self._select_feature_to_split(sample, output)
        feature_name = feature_label[feature_to_split]
        sub_node = _TreeNote(feature_name, feature_to_split)
        if self.root is None:
            self.root = sub_node
        else:
            node.add_subtree(current_feature_value, sub_node)
        unique_feature_value = np.unique(sample[:, feature_to_split])

        for feature_value in unique_feature_value:
            sub_sample, sub_output, sub_feature_labels = self._split_data_set(sample, output, feature_to_split,
                                                                              feature_value, feature_label)
            self._train(sub_sample, sub_output, feature_value, sub_feature_labels, sub_node)
