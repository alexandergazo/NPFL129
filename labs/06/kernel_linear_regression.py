#!/usr/bin/env python3
import argparse
import sys

import numpy as np
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--iterations", default=200, type=int, help="Number of training iterations")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")


# If you add more arguments, ReCodEx will keep them with your default values.


def main(args):
    def kernel(x, y):
        if args.kernel == 'rbf':
            return np.exp(-args.kernel_gamma * np.linalg.norm(x - y) ** 2)
        elif args.kernel == 'poly':
            d, r = args.kernel_gamma * x * y + 1, 1
            for _ in range(args.kernel_degree): r *= d
            return r

    def predict(X):
        predictions = np.empty((X.shape[0], 1))
        for i, row in enumerate(X):
            predictions[i] = sum(beta * kernel(row, dictum) for beta, dictum in zip(betas, train_data))
        return predictions

    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1

    betas = np.zeros(args.data_size)

    train_rmses, test_rmses = [], []
    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        for i in range(train_data.shape[0] // args.batch_size):
            batch_indices = permutation[i * args.batch_size:(i + 1) * args.batch_size]
            betas[batch_indices] = betas[batch_indices] + args.learning_rate / len(batch_indices) * (
                        train_target[batch_indices] - predict(train_data[batch_indices]).flatten())

        train_rmses.append(mean_squared_error(train_target, predict(train_data), squared=False))
        test_rmses.append(mean_squared_error(test_target, predict(test_data), squared=False))

        if (iteration + 1) % 10 == 0:
            print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
                iteration + 1, train_rmses[-1], test_rmses[-1]))

    if args.plot:
        import matplotlib.pyplot as plt
        test_predictions = predict(test_data)

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")

        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        if args.plot is True:
            plt.show()
        else:
            plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return train_rmses, test_rmses


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
