#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()

# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--range", default=9, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=40, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


def main(args):
    # Create the data
    xs = np.linspace(0, 7, num=args.data_size)
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)
    xs = xs.reshape((xs.shape[0], 1))

    rmses = []
    data = np.empty([xs.shape[0], 0])
    for order in range(1, args.range + 1):
        data = np.append(data, xs ** order, axis=1)

        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, ys, test_size=args.test_size, random_state=args.seed)

        model = sklearn.linear_model.LinearRegression().fit(train_data, train_target)
        predict = model.predict(test_data)

        rmse = np.sqrt(sklearn.metrics.mean_squared_error(predict, test_target))
        rmses.append(rmse)

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                if not plt.gcf().get_axes(): plt.figure(figsize=(6.4*3, 4.8*3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.plot(train_data[:, 0], train_target, "go")
            plt.plot(test_data[:, 0], test_target, "ro")
            plt.plot(np.linspace(xs[0], xs[-1], num=100),
                     model.predict(np.stack([np.linspace(xs[0], xs[-1], num=100)**order for order in range(1, order + 1)], axis=1)), "b")
            if args.plot is True: plt.show()
            else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return rmses


if __name__ == "__main__":
    args = parser.parse_args()
    rmses = main(args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))
