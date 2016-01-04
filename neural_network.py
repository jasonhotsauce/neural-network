import gzip
import os
import numpy as np
import urllib

from numpy.linalg import linalg

SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"


def download_if_needed(filename, directory):
    """

    :rtype: string
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully Downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _restruct_labels(labels, num_classes=10):
    num_labels = labels.shape[0]
    offset = np.arange(num_labels) * num_classes
    labels_vector = np.zeros((num_labels, num_classes))
    labels_vector.flat[offset + labels.ravel()] = 1
    return labels_vector


def read_image_data(filename):
    print('Extracting Image Data', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('ERROR')
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)

        return _normalize_data(data)


def _normalize_data(data):
    data_norm = np.zeros(data.shape)
    mu = np.zeros((1, data.shape[1]))
    sigma = np.zeros((1, data.shape[1]))

    for i in range(data.shape[1]):
        mu[0, i] = np.mean(data[:, i])
        sigma[0, i] = np.std(data[:, i])
        if sigma[0, i] == 0:
            data_norm[:, i] = data[:, i]
        else:
            data_norm[:, i] = (data[:, i] - mu[0, i]) / sigma[0, i]
    return data_norm


def read_label_data(filename):
    print('Extracting Label Data', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid Label data magic Number: %d in MNIST data: %s', (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return _restruct_labels(labels)


def rand_initialize_weights(L_in, L_out):
    init_epsilon = np.sqrt(6) / (L_in + L_out)
    weights = np.random.rand(L_out, 1 + L_in) * 2 * init_epsilon - init_epsilon
    return weights


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def predict(input, theta1, theta2):
    transfered_input = np.c_[np.ones(input.shape[0]), input]
    h1 = sigmoid(np.dot(transfered_input, theta1.T))
    h2 = sigmoid(np.dot(np.c_[np.ones(h1.shape[0]), h1], theta2.T))
    print(h2[0, :])
    return h2.argmax(axis=1).reshape(h2.shape[0], 1)


def cost_function(nn_params, input_layer_size, hidden_layer_size, input, output, lambda_num, num_labels=10):
    num_samples = input.shape[0]
    transfered_input = np.c_[np.ones(num_samples), input]
    theta1 = nn_params[:, 0:hidden_layer_size * (1 + input_layer_size)].reshape(hidden_layer_size, 1 + input_layer_size)
    theta2 = nn_params[:, hidden_layer_size * (1 + input_layer_size):].reshape(num_labels, 1 + hidden_layer_size)
    # Feed Forward
    z2 = np.dot(transfered_input, theta1.T)
    layer = sigmoid(z2)
    transfered_layer = np.c_[np.ones(num_samples), layer]
    z3 = np.dot(transfered_layer, theta2.T)
    hypothesis = sigmoid(z3)

    # calculate cost
    cost = (
               -1 * output * np.log(hypothesis) - (1 - output) * np.log(
                       (1 - hypothesis))).sum() / num_samples + lambda_num * (
        np.power(theta1[:, 1:], 2).sum() + np.power(theta2[:, 1:], 2).sum()) / (2 * num_samples)

    # Back Propagation.
    delta3 = hypothesis - output
    delta2 = np.dot(delta3, theta2[:, 1:]) * sigmoid_gradient(z2)
    delta2_matrix = np.dot(transfered_layer.T, delta3).T
    delta1_matrix = np.dot(transfered_input.T, delta2).T

    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    theta2_grad[:, 0] = delta2_matrix[:, 0] / num_samples
    theta2_grad[:, 1:] = delta2_matrix[:, 1:] / num_samples + (lambda_num / num_samples) * theta2[:, 1:]
    theta1_grad[:, 0] = delta1_matrix[:, 0] / num_samples
    theta1_grad[:, 1:] = delta1_matrix[:, 1:] / num_samples + (lambda_num / num_samples) * theta1[:, 1:]
    return cost, unroll_params(theta1_grad, theta2_grad)


def generate_debug_input(num_samples, num_features):
    w = np.zeros((num_samples, 1 + num_features))
    w = np.sin(np.arange(w.size) + 1).reshape(w.shape) / 10
    return w


def varify_gradient_decent(lambda_num):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    input = generate_debug_input(m, input_layer_size - 1)
    print(input)
    output = 1 + np.mod(np.arange(m).reshape(1, m), num_labels).T
    print(output)
    theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    nn_param = unroll_params(theta1, theta2)

    cost_func = lambda param: cost_function(param, input_layer_size, hidden_layer_size, input, output, lambda_num,
                                            num_labels)
    (cost, grad) = cost_func(nn_param)
    num_grad = compute_numerical_gradient(cost_func, nn_param)
    diff = linalg.norm(num_grad - grad, 2) / linalg.norm(num_grad + grad,2)
    print("The relative difference will be small (less than 1e-9)", diff)


def compute_numerical_gradient(cost_func, theta):
    print(theta)
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[0, p] = e
        (loss1, _) = cost_func(theta - perturb)
        (loss2, _) = cost_func(theta + perturb)
        # Compute Numerical Gradient
        numgrad[0, p] = (loss2 - loss1) / (2 * e)
        perturb[0, p] = 0
    return numgrad


def unroll_params(theta1, theta2):
    return np.concatenate((theta1.reshape(1, theta1.shape[0] * theta1.shape[1]),
                           theta2.reshape(1, theta2.shape[0] * theta2.shape[1])), axis=1)


TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

file_path = download_if_needed(TRAIN_IMAGES, "./test/")
image_data = read_image_data(file_path)
label_path = download_if_needed(TRAIN_LABELS, "./test/")
labels = read_label_data(label_path)

test_file_path = download_if_needed(TEST_IMAGES, "./test/")
test_image_data = read_image_data(test_file_path)
test_label_path = download_if_needed(TEST_LABELS, "./test/")
test_label_data = read_label_data(test_label_path)

m = image_data.shape[0]

input_layer_size = image_data.shape[1]
hidden_layer_size = 15
num_labels = 10

theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

learning_rate = 3.0
lambda_num = 3

# varify_gradient_decent(lambda_num)

for iteration in range(400):
    nn_param = unroll_params(theta1, theta2)
    (J, grad) = cost_function(nn_param, input_layer_size, hidden_layer_size, image_data, labels, lambda_num)
    theta1_grad = grad[:, 0:hidden_layer_size * (1 + input_layer_size)].reshape(hidden_layer_size, 1 + input_layer_size)
    theta2_grad = grad[:, hidden_layer_size * (1 + input_layer_size):].reshape(num_labels, 1 + hidden_layer_size)
    theta1 -= learning_rate * theta1_grad
    theta2 -= learning_rate * theta2_grad
    print("Iteration: %d | Loss: %f" % (iteration, J))

h = predict(test_image_data, theta1, theta2)
print("Hypothesis: ", h)
r = test_label_data.argmax(axis=1).reshape(test_label_data.shape[0], 1)
print("Real Label: ", r)
print(np.equal(h, r).astype(np.uint32))
accuracy = np.mean(np.equal(h, r).astype(np.uint32)) * 100

print("Predicting Accuracy ", accuracy, "%")
