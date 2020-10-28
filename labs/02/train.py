#!/usr/bin/env python3
#
# All team solutions **must** list **all** members of the team.
# The members must be listed using their ReCodEx IDs anywhere
# in a comment block in the source file (on a line beginning with `#`).
#
# You can find out ReCodEx ID in the URL bar after navigating
# to your User profile page. The ID has the following format:
# 02d8c0c3-3972-11e8-9b58-00505601122b

import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from LinearRegressionSGD import LinearRegressionSGD


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: sprint, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rentals in the given hour.
    """

    def __init__(self,
                 name="rental_competition.train.npz",
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
parser.add_argument("--seed", default=32, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


def main(args):
    if os.path.isfile(args.model_path):
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
    else:
        dataset = Dataset()
        dataset.data = np.append(dataset.data, np.ones([dataset.data.shape[0], 1]), axis=1)
        ints = np.all(dataset.data.astype(int) == dataset.data, axis=0)
        ct = ColumnTransformer([("OneHot", OneHotEncoder(sparse=False, handle_unknown="ignore"), ints),
                                ("Normalize", StandardScaler(), ~ints)])
        model = Pipeline([
            ("OH and Normalize", ct),
            ("PF", PolynomialFeatures(3, include_bias=False)),
            ("LR", LinearRegressionSGD(200, 20, 0.01, 0.003, args.seed))
            # PARAMS FOUND VIA rental_competition.py --predict="gridCV"
        ])
        model.fit(dataset.data, dataset.target)
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    dataset = Dataset(args.predict)
    dataset.data = np.append(dataset.data, np.ones([dataset.data.shape[0], 1]), axis=1)

    predictions = model.predict(dataset.data)

    return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
