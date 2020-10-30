#!/usr/bin/env python3
import argparse
from collections import defaultdict

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--iterations", default=10, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--verbose", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.


class LinearLayer:
    def __init__(self, n_inputs, n_units, name, weights=None, biases=None):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.name = name
        self.W = weights
        self.b = biases

    def has_params(self):
        return True

    def forward(self, X):
        assert X.shape[1] == self.n_inputs
        return X @ self.W + self.b

    def delta(self, Y, delta_next):
        assert Y.shape == delta_next.shape
        assert Y.shape[1] == self.n_units
        return delta_next @ self.W.transpose()

    def grad(self, X, delta_next):
        assert X.shape[1] == self.n_inputs
        assert delta_next.shape[1] == self.n_units
        dW = X.transpose() @ delta_next
        dW /= X.shape[0]
        db = np.mean(delta_next, axis=0)
        assert db.shape == self.b.shape, "{} {}".format(db.shape, self.b.shape)
        assert dW.shape == self.W.shape, dW.shape + self.W.shape
        return [dW, db]

    def update_params(self, dtheta):
        assert len(dtheta) == 2, len(dtheta)
        dW, db = dtheta
        assert dW.shape == self.W.shape, dW.shape
        assert db.shape == self.b.shape, db.shape
        self.W += dW
        self.b += db

    def initialize(self):
        pass


class ReLULayer:
    def __init__(self, name):
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        # X[X < 0] = 0  # best performance but modifies X
        XX = X * (X > 0)
        return XX

    def delta(self, Y, delta_next):
        YS = (Y > 0)
        return YS * delta_next


def rowsoftmax(X):
    max = np.max(X, axis=-1, keepdims=True)
    numerator = np.exp(X - max)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator


class SoftmaxLayer:
    def __init__(self, name):
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        return rowsoftmax(X)

    def delta(self, Y, delta_next):
        assert Y.shape == delta_next.shape
        return (-np.sum(delta_next * Y, axis=1, keepdims=True) + delta_next) * Y


class MLP:
    def __init__(self, n_inputs, layers, loss, output_layers=[]):
        self.n_inputs = n_inputs
        self.layers = layers
        self.output_layers = output_layers
        self.loss = loss
        self.first_param_layer = layers[-1]
        for l in layers:
            if l.has_params():
                self.first_param_layer = l
                break

    def propagate(self, X, output_layers=True, last_layer=None):
        layers = self.layers + (self.output_layers if output_layers else [])
        if last_layer is not None:
            layer_names = [layer.name for layer in layers]
            layers = layers[0: layer_names.index(last_layer) + 1]
        for layer in layers:
            X = layer.forward(X)
        return X

    def evaluate(self, X, T):
        return self.loss.forward(self.propagate(X, output_layers=False), T)

    def gradient(self, X, T):
        Ys = []
        message = X
        for layer in self.layers:
            message = layer.forward(message)
            Ys.append(message)
        g = {}
        Y = Ys[-1]
        d = self.loss.delta(Y, T)
        idx = len(self.layers) - 1
        for layer in reversed(self.layers):
            Y = Ys[idx]
            if layer.has_params():
                if idx == 0:
                    YY = X
                else:
                    YY = Ys[idx - 1]
                g[layer.name] = layer.grad(YY, d)
            d = layer.delta(Y, d)
            idx -= 1
        return g


class LossCrossEntropy:
    def __init__(self, name):
        self.name = name

    def forward(self, X, T):
        assert X.shape == T.shape
        return np.sum(-np.log(X) * T, axis=-1, keepdims=True)

    def delta(self, X, T):
        assert X.shape == T.shape
        return - T / X


def accuracy(Y, T):
    p = np.argmax(Y, axis=1)
    t = np.argmax(T, axis=1)
    return np.mean(p == t)


def train(net, train_data, train_target, batch_size=1, epochs=2, learning_rate=0.1, test_data=None, test_target=None,
          verbose=False, generator=None):
    """
    Trains a network using vanilla gradient descent.
    """
    n_samples = train_data.shape[0]
    assert train_target.shape[0] == n_samples
    assert batch_size <= n_samples
    run_info = defaultdict(list)

    def process_info(epoch):
        Y = net.propagate(train_data)
        train_loss = net.loss.forward(Y, train_target)
        train_accuracy = accuracy(Y, train_target)
        run_info['train_loss'].append(train_loss)
        run_info['train_accuracy'].append(train_accuracy)
        if test_data is not None:
            Y = net.propagate(test_data)
            test_loss = net.loss.forward(Y, test_target)
            test_accuracy = accuracy(Y, test_target)
            run_info['test_loss'].append(test_loss)
            run_info['test_accuracy'].append(test_accuracy)

    process_info('initial')
    for epoch in range(1, epochs + 1):
        permutation = generator.permutation(train_data.shape[0]) if generator is not None else list(
            range(train_data.shape[0]))
        offset = 0
        while offset < n_samples:
            last = min(offset + batch_size, n_samples)
            grads = net.gradient(np.asarray(train_data[permutation[offset:last]]),
                                 np.asarray(train_target[permutation[offset:last]]))
            for layer in net.layers:
                if layer.has_params():
                    gs = grads[layer.name]
                    dtheta = [-learning_rate * g for g in gs]
                    layer.update_params(dtheta)
            offset += batch_size
        process_info(epoch)
        if verbose:
            print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
                epoch, 100 * run_info['train_accuracy'][-1], 100 * run_info['test_accuracy'][-1]))
    return run_info


def one_hot(classes, x):
    result = np.zeros((len(x), classes))
    result[list(range(len(x))), x] = 1
    return result


def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)
    test_target, train_target = one_hot(args.classes, test_target), one_hot(args.classes, train_target)

    # Generate initial model weights
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    net = MLP(
        n_inputs=train_data.shape[0],
        layers=[
            LinearLayer(n_inputs=train_data.shape[1], n_units=args.hidden_layer, name='Linear_1', weights=weights[0],
                        biases=biases[0]),
            ReLULayer(name='ReLU_1'),
            LinearLayer(n_inputs=args.hidden_layer, n_units=args.classes, name='Linear_2', weights=weights[1],
                        biases=biases[1]),
            SoftmaxLayer(name='Softmax_OUT')
        ],
        loss=LossCrossEntropy(name='CE')
    )
    train(net, train_data, train_target, args.batch_size, args.iterations, args.learning_rate, test_data, test_target,
          args.verbose, generator)

    return tuple(weights + biases)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters = main(args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:20]] + ["..."]) for ws in parameters), sep="\n")
