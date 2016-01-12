import unittest
import numpy as np
from decision_tree import DecisionTree


class DecisionTreeTests(unittest.TestCase):

    def setUp(self):
        self.sample_input, self.sample_output, self.feature_labels = self._create_test_data()

    def tearDown(self):
        del self.sample_output
        del self.sample_input
        del self.feature_labels

    def _create_test_data(self):
        input_val = np.array([[1, 1], [1, 1], [0, 1], [1, 0], [0, 1]])
        output = ['yes', 'yes', 'no', 'no', 'no']
        input_labels = ['no surfacing', 'flippers']
        return input_val, output, input_labels

    def test_calculate_shannon_entropy(self):
        h = DecisionTree._calculate_shannon_entropy(self.sample_output)
        self.assertAlmostEqual(h, 0.970951, places=5, msg='Shannon entropy should be 0.970951, but get: %f' % h)

    def test_split_data(self):
        new_sample, new_output, sub_feature_list = DecisionTree._split_data_set(self.sample_input, self.sample_output,
                                                                                0, 1, self.feature_labels)
        np.testing.assert_array_equal(new_sample, np.array([[1], [1], [0]]))
        self.assertListEqual(new_output, ['yes', 'yes', 'no'])
        self.assertListEqual(sub_feature_list, ['flippers'])

    def test_choose_feature_to_split(self):
        decision_tree = DecisionTree()
        feature_to_split = decision_tree._select_feature_to_split(self.sample_input, self.sample_output)
        self.assertEqual(feature_to_split, 0, 'The best feature index to pick is 0, but get %d' % feature_to_split)

    def test_train_method(self):
        decision_tree = DecisionTree()
        decision_tree.train(self.sample_input, self.sample_output, feature_label=self.feature_labels)
        self.assertIsNotNone(decision_tree.root, 'Decision tree must have a root node')

    def test_decision_tree(self):
        decision_tree = DecisionTree()
        decision_tree.train(self.sample_input, self.sample_output, feature_label=self.feature_labels)
        self.assertEqual(decision_tree.root.value, 'no surfacing')
        test_input = [1, 1]
        test_output = decision_tree.predict(test_input)
        self.assertEqual(test_output, 'yes')

if __name__ == '__main__':
    unittest.main()
