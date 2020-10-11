#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    dataset.feature_names = np.append(dataset.feature_names, ' ')
    dataset.data = np.append(dataset.data, np.ones([dataset.data.shape[0], 1]), axis=1)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data,
                                                                                                dataset.target,
                                                                                                test_size=args.test_size,
                                                                                                random_state=args.seed)

    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_data.T, train_data)), train_data.T), train_target)

    predict = np.matmul(test_data, w)

    rmse = np.sqrt(mean_squared_error(predict, test_target))
    return rmse


if __name__ == "__main__":
    args = parser.parse_args()
    rmse = main(args)
    print("{:.2f}".format(rmse))
