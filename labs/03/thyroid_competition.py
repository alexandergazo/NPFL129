#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """

    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        train.data = np.append(train.data, np.ones((train.data.shape[0], 1)), axis=1)
        pipeline = Pipeline([
            ("CT", ColumnTransformer([("SS", StandardScaler(), slice(15, 21))], remainder='passthrough')),
            ("PF", PolynomialFeatures(include_bias=False)),
            ("LR", LogisticRegression(fit_intercept=False, class_weight='balanced', random_state=args.seed, max_iter=3000))
        ])
        trainer = GridSearchCV(
            pipeline,
            {'LR__C': np.geomspace(2000, 5000, 50), 'PF__degree': [2, 3],
             'LR__solver': ['newton-cg', 'lbfgs', 'liblinear']},
            cv=StratifiedKFold(5), verbose=10, n_jobs=8
        )
        trainer.fit(train.data, train.target)

        model = trainer

        print(model.best_score_)
        print(model.best_params_)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)
        test.data = np.append(test.data, np.ones((test.data.shape[0], 1)), axis=1)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        print(model.best_score_)
        print(model.best_params_)
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
