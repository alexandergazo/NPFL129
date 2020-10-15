#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    generator = np.random.RandomState(args.seed)
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)
    data = np.append(data, np.ones([data.shape[0], 1]), axis=1)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target,
                                                                                                test_size=args.test_size,
                                                                                                random_state=args.seed)
    weights = generator.uniform(size=train_data.shape[1])

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        for i in range(train_data.shape[0] // args.batch_size):
            batch_data = train_data[permutation[i * args.batch_size:(i + 1) * args.batch_size]]
            batch_target = train_target[permutation[i * args.batch_size:(i + 1) * args.batch_size]]
            batch_prediction = np.matmul(batch_data, weights)
            gradient = np.matmul(batch_data.T, (batch_prediction - batch_target)) / args.batch_size
            weights = weights - args.learning_rate * gradient

        train_prediction = np.matmul(train_data, weights)
        test_prediction = np.matmul(test_data, weights)
        train_rmses.append(sklearn.metrics.mean_squared_error(train_prediction, train_target, squared=False))
        test_rmses.append(sklearn.metrics.mean_squared_error(test_prediction, test_target, squared=False))

    explicit_prediction = sklearn.linear_model.LinearRegression().fit(train_data, train_target).predict(test_data)
    explicit_rmse = sklearn.metrics.mean_squared_error(explicit_prediction, test_target, squared=False)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.legend()
        if args.plot is True:
            plt.show()
        else:
            plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.2f}, explicit {:.2f}".format(sgd_rmse, explicit_rmse))
