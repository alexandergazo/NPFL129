#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--iterations", default=50, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size,
                                                                        random_state=args.seed)
    weights = generator.uniform(size=train_data.shape[1])

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        for i in range(train_data.shape[0] // args.batch_size):
            batch_data = train_data[permutation[i * args.batch_size:(i + 1) * args.batch_size]]
            batch_target = train_target[permutation[i * args.batch_size:(i + 1) * args.batch_size]]
            gradients = - (batch_target - sigmoid(batch_data @ weights)).reshape(-1, 1) * batch_data
            weights -= args.learning_rate * gradients.mean(axis=0)

        train_accuracy = accuracy_score(train_target, sigmoid(train_data @ weights) > 1 / 2)
        train_loss = log_loss(train_target, sigmoid(train_data @ weights))
        test_accuracy = accuracy_score(test_target, sigmoid(test_data @ weights) > 1 / 2)
        test_loss = log_loss(test_target, sigmoid(test_data @ weights))

        print("After iteration {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            iteration + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                if not iteration: plt.figure(figsize=(6.4 * 3, 4.8 * (args.iterations + 2) // 3))
                plt.subplot(3, (args.iterations + 2) // 3, 1 + iteration)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=21, cmap=plt.cm.RdBu, alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="P", label="train", cmap=plt.cm.RdBu)
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap=plt.cm.RdBu)
            plt.legend(loc="upper right")
            if args.plot is True:
                plt.show()
            else:
                plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
