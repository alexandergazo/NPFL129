#!/usr/bin/env python3
import argparse
import sys

from scipy.special import softmax
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from mlp import *

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--iterations", default=10, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")

# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--verbose", default=False, action="store_true", help="Print training progress.")


def one_hot(classes, x):
    result = np.zeros((len(x), classes))
    result[list(range(len(x))), x] = 1
    return result


def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    # data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)
    test_target, train_target = one_hot(args.classes, test_target), one_hot(args.classes, train_target)

    # Generate initial linear regression weights
    weights = generator.uniform(size=[train_data.shape[1] + 1, args.classes], low=-0.1, high=0.1)
    w = weights[:-1, :]
    b = weights[-1, :]
    print(weights.shape, w.shape, b.shape)

    net = MLP(train_data.shape[1],
              [LinearLayer(train_data.shape[1], args.classes, "Linear", w, b),
               SoftmaxLayer("Softmax")],
              loss=LossCrossEntropy(name='CE'))
    train(net, train_data, train_target, args.batch_size, args.iterations, args.learning_rate, test_data, test_target,
          args.verbose, generator)

    return weights


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
