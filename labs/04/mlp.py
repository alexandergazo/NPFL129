from collections import defaultdict
import numpy as np


class LinearLayer:
    def __init__(self, n_inputs, n_units, name, weights=None, biases=None):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.name = name
        self.W = weights
        self.b = biases
        if weights is None:
            self.initialize()

    def has_params(self):
        return True

    def forward(self, X):
        assert X.shape[1] == self.n_inputs
        return X @ self.W + (self.b if self.b is not None else 0)

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
        if self.b is not None: assert db.shape == self.b.shape, "{} {}".format(db.shape, self.b.shape)
        assert dW.shape == self.W.shape, dW.shape + self.W.shape
        return [dW, db]

    def update_params(self, dtheta):
        assert len(dtheta) == 2, len(dtheta)
        dW, db = dtheta
        assert dW.shape == self.W.shape, dW.shape
        if self.b is not None:
            assert db.shape == self.b.shape, db.shape
            self.b += db
        self.W += dW

    def initialize(self):
        """
        Perform He's initialization (https://arxiv.org/pdf/1502.01852.pdf). This method is tuned for ReLU activation
        function. Biases are initialized to 1 increasing probability that ReLU is not initially turned off.
        """
        scale = np.sqrt(2.0 / self.n_inputs)
        self.W = np.random.normal(loc=0.0, scale=scale, size=(self.n_inputs, self.n_units))
        self.b = np.ones(self.n_units)


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
        return lossCrossEntropyFunction(X, T)

    def delta(self, X, T):
        assert X.shape == T.shape
        return - T / X


def lossCrossEntropyFunction(X, T):
    log = -np.log(X)
    return np.sum(log * T, axis=-1, keepdims=True)


class LossCrossEntropyForSoftmaxLogits:
    def __init__(self, name):
        self.name = name

    def forward(self, X, T):
        assert X.shape == T.shape
        XX = rowsoftmax(X)
        return lossCrossEntropyFunction(XX, T)

    def delta(self, X, T):
        assert X.shape == T.shape
        Y = rowsoftmax(X)
        return - T + Y * np.sum(T, axis=1, keepdims=True)


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
            print("After iteration {}: train acc {:.1f}%".format(epoch, 100 * run_info['train_accuracy'][-1]), end="")
            print(", test acc {:.1f}%".format(100 * run_info['test_accuracy'][-1]) if test_data is not None else "")
    return run_info
