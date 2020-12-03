#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.base
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.svm

class MNIST:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=0.022, type=float, help="RBF gamma")
parser.add_argument("--max_iter", default=100, type=int, help="Maximum iterations for LR")
parser.add_argument("--nystroem", default=0, type=int, help="Use Nystroem approximation")
parser.add_argument("--original", default=False, action="store_true", help="Use original data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--rff", default=0, type=int, help="Use RFFs")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--svm", default=False, action="store_true", help="Use SVM instead of LR")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

class RFFsTransformer(sklearn.base.TransformerMixin):
    def __init__(self, n_components, gamma, seed):
        self._n_components = n_components
        self._gamma = gamma
        self._seed = seed

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        self._w = np.sqrt(2 * self._gamma) * generator.normal(size=(X.shape[1], self._n_components))
        self._b = generator.uniform(0, 2 * np.pi, size=(self._n_components,))

        return self

    def transform(self, X):
        return np.sqrt(2 / self._n_components) * np.cos(X @ self._w + self._b)

class NystroemTransformer(sklearn.base.TransformerMixin):
    def __init__(self, n_components, gamma, seed):
        self._n_components = n_components
        self._gamma = gamma
        self._seed = seed

    def _rbf_kernel(self, X, Z):
        kernel = np.empty((X.shape[0], Z.shape[0]))
        for i, x in enumerate(X):
            kernel[i] = np.exp(-self._gamma * np.linalg.norm(Z - x, axis=1) ** 2)
        return kernel

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        indices = generator.choice(X.shape[0], size=self._n_components, replace=False)

        K = self._rbf_kernel(X[indices], X[indices])

        U, D, V = np.linalg.svd(K)
        D_inv = np.power(D + 1e-12, -1/2)

        self._training_subset = X[indices]
        self._V = (U * D_inv) @ V

        return self

    def transform(self, X):
        K = self._rbf_kernel(X, self._training_subset)
        return K @ self._V

def main(args):
    # Use the digits dataset.
    dataset = MNIST(data_size=5000)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    features = []
    if args.original:
        features.append(("original", sklearn.preprocessing.FunctionTransformer()))
    if args.rff:
        features.append(("rff", RFFsTransformer(args.rff, args.gamma, args.seed)))
    if args.nystroem:
        features.append(("nystroem", NystroemTransformer(args.nystroem, args.gamma, args.seed)))

    if args.svm:
        classifier = sklearn.svm.SVC()
    else:
        classifier = sklearn.linear_model.LogisticRegression(solver="saga", penalty="none", max_iter=args.max_iter, random_state=args.seed)

    pipeline = sklearn.pipeline.Pipeline([
        ("scaling", sklearn.preprocessing.MinMaxScaler()),
        ("features", sklearn.pipeline.FeatureUnion(features)),
        ("classifier", classifier),
    ])
    pipeline.fit(train_data, train_target)

    test_accuracy = sklearn.metrics.accuracy_score(test_target, pipeline.predict(test_data))
    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
