#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge


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
parser.add_argument("--seed", default=None, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
parser.add_argument("--dev_size", default="0.15", type=float, help="Development set size")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--seed_count", default=1, type=int, help="Seed count")


def main(args):
    if args.predict is None:
        dataset = Dataset()
        dataset.data = np.append(dataset.data, np.ones([dataset.data.shape[0], 1]), axis=1)
        alphas = np.linspace(1, 70, num=100)
        rmses = np.empty((args.seed_count, len(alphas)))

        ints = np.all(dataset.data.astype(int) == dataset.data, axis=0)
        ct = ColumnTransformer([
            ("OneHot", OneHotEncoder(sparse=False, handle_unknown="ignore"), ints),
            ("Normalize", StandardScaler(), ~ints)
        ])
        pf = PolynomialFeatures(3, include_bias=False)

        models = []
        for alpha in alphas:
            model = Pipeline([
                ("Feature Preprocessing", ct),
                ("Polynomial Features 3rd degree", pf),
                ("Ridge Model alpha={}".format(alpha), Ridge(alpha))
            ], verbose=True)
            models.append(model)

        trained_models = []
        for i in range(args.seed_count):
            np.random.seed(args.seed)
            train_data, dev_data, train_target, dev_target = train_test_split(dataset.data, dataset.target,
                                                                              test_size=args.dev_size,
                                                                              random_state=args.seed)
            trained_models = list(map(lambda model: model.fit(train_data, train_target), models))
            predictions = map(lambda model: model.predict(dev_data), trained_models)
            rmses_tmp = map(lambda prediction: mean_squared_error(prediction, dev_target, squared=False), predictions)
            rmses[i, :] = np.asarray(list(rmses_tmp))

        mean_rmses = rmses.mean(axis=0)
        std_rmses = rmses.std(axis=0)
        best_alpha = alphas[mean_rmses.argmin()]
        best_model = trained_models[mean_rmses.argmin()]
        best_rmse = mean_rmses.min()

        print("Best result", best_rmse, "achieved with alpha =", best_alpha)

        if args.plot:
            import matplotlib.pyplot as plt
            plt.errorbar(alphas, mean_rmses, yerr=std_rmses)
            plt.xscale("log")
            plt.xlabel("L2 regularization strength")
            plt.ylabel("RMSE")
            args.plot = "RMSE_lambda"
            if args.plot is True:
                plt.show()
            else:
                plt.savefig(args.plot, transparent=True, bbox_inches="tight")
                plt.savefig(args.plot + "2", transparent=False, bbox_inches="tight")
                plt.show()

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(best_model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        dataset = Dataset(args.predict)
        dataset.data = np.append(dataset.data, np.ones([dataset.data.shape[0], 1]), axis=1)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(dataset.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
