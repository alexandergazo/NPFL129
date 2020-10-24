#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data,
                                                                                                dataset.target,
                                                                                                test_size=args.test_size,
                                                                                                random_state=args.seed)
    pipeline = Pipeline([
        ("MMS", MinMaxScaler()),
        ("PF", PolynomialFeatures()),
        ("LR", LogisticRegression(random_state=args.seed))
    ])
    trainer = GridSearchCV(
        pipeline,
        {'LR__C': [0.01, 1, 100], 'PF__degree': [1, 2], 'LR__solver': ['lbfgs', 'sag']},
        cv=StratifiedKFold(5)
    )
    trainer.fit(train_data, train_target)

    return accuracy_score(trainer.predict(test_data), test_target)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))
