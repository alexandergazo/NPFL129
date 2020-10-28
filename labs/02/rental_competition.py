#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, SGDRegressor
import wandb
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
parser.add_argument("--dev_size", default=0.10, type=float, help="Development set size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--batch_size", default=20, type=int, help="Batch size")
parser.add_argument("--epochs", default=200, type=int, help="Number of SGD iterations over the data")
parser.add_argument("--learning_rate", default=0.003, type=float, help="Learning rate")
parser.add_argument("--alpha", default=0.01, type=int, help="L2 rate")


def main(args):
    generator = np.random.RandomState(args.seed)
    if args.predict == "gridCV":
        # wandb.init(project="npfl129", name="GridCV")
        np.random.seed(args.seed)
        dataset = Dataset()
        dataset.data = np.append(dataset.data, np.ones([dataset.data.shape[0], 1]), axis=1)
        train_data, dev_data, train_target, dev_target = train_test_split(dataset.data, dataset.target,
                                                                          test_size=args.dev_size,
                                                                          random_state=args.seed)
        ints = np.all(train_data.astype(int) == train_data, axis=0)
        ct = ColumnTransformer([("OneHot", OneHotEncoder(sparse=False, handle_unknown="ignore"), ints),
                                ("Normalize", StandardScaler(), ~ints)])
        feature_processing = Pipeline([
            ("OH and Normalize", ct),
            ("PF", PolynomialFeatures(3, include_bias=False))
        ])
        trainer = GridSearchCV(
            LinearRegressionSGD(random_state=args.seed, epochs=60),
            {'alpha': np.geomspace(1, 50, 30),
             'learning_rate': np.geomspace(0.001, 0.1, 30),
             'batch_size': [20, 50, 100, 200, 300]},
            cv=KFold(5), verbose=3, n_jobs=2
        )
        trainer.fit(feature_processing.fit_transform(train_data), train_target)
        model = Pipeline([("FP", feature_processing), ("LR", trainer)])
        train_rmse = mean_squared_error(model.predict(train_data), train_target, squared=False)
        print(train_rmse)
        dev_rmse = mean_squared_error(model.predict(dev_data), dev_target, squared=False)
        print(dev_rmse)
        for epoch in range(args.epochs):
            wandb.log({"train_rmse": train_rmse})
            wandb.log({"dev_rmse": dev_rmse})
        with lzma.open(args.model_path + "GR", "wb") as model_file:
            pickle.dump(model, model_file)
        with lzma.open("trainer", "wb") as trainer_file:
            pickle.dump(trainer, trainer_file)
    elif args.predict == "ridgeCV":
        wandb.init(project="npfl129", name="SGD_REGRESSOR")
        np.random.seed(args.seed)
        dataset = Dataset()

        dataset.data = np.append(dataset.data, np.ones([dataset.data.shape[0], 1]), axis=1)
        train_data, dev_data, train_target, dev_target = train_test_split(dataset.data, dataset.target,
                                                                          test_size=args.dev_size,
                                                                          random_state=args.seed)
        ints = np.all(train_data.astype(int) == train_data, axis=0)
        ct = ColumnTransformer([("OneHot", OneHotEncoder(sparse=False, handle_unknown="ignore"), ints),
                                ("Normalize", StandardScaler(), ~ints)])
        pf = PolynomialFeatures(3, include_bias=False)
        alphas = np.geomspace(0.1, 70, 200)
        pipelines = [Pipeline([
            ("Feature Preprocessing", ct),
            ("Polynomial Features 3rd degree", pf),
            ("Linear Model", SGDRegressor(alpha=alpha))
        ]) for alpha in alphas]
        models = map(lambda m: m.fit(train_data, train_target), pipelines)
        losses = map(lambda model: mean_squared_error(model.predict(dev_data), dev_target, squared=False), models)
        # wandb.log({"Alphas SGD": wandb.plot.line(
        #     wandb.Table(data=[[x, y] for x, y in zip(alphas, losses)], columns=["x", "y"]), "x", "y")})
        for alpha, loss in zip(alphas, losses):
            wandb.log({"Loss": loss, "Alpha": alpha})
    elif args.predict == "ridge":
        wandb.init(project="npfl129", name="RIDGE BASELINE")
        np.random.seed(args.seed)
        dataset = Dataset()

        dataset.data = np.append(dataset.data, np.ones([dataset.data.shape[0], 1]), axis=1)
        train_data, dev_data, train_target, dev_target = train_test_split(dataset.data, dataset.target,
                                                                          test_size=args.dev_size,
                                                                          random_state=args.seed)
        ints = np.all(train_data.astype(int) == train_data, axis=0)
        ct = ColumnTransformer([("OneHot", OneHotEncoder(sparse=False, handle_unknown="ignore"), ints),
                                ("Normalize", StandardScaler(), ~ints)])
        pf = PolynomialFeatures(3, include_bias=False)
        pipeline = Pipeline([
            ("Feature Preprocessing", ct),
            ("Polynomial Features 3rd degree", pf),
            ("Linear Model", Ridge(alpha=10))
        ])
        model = pipeline.fit(train_data, train_target)
        loss = mean_squared_error(model.predict(dev_data), dev_target, squared=False)
        for epoch in range(args.epochs):
            wandb.log({"dev_rmse": loss})
    elif args.predict is None:
        wandb.init(project="npfl129", name="init_rental_alpha_competition")
        wandb.config.update(args)
        dataset = Dataset()
        np.random.seed(args.seed)

        dataset.data = np.append(dataset.data, np.ones([dataset.data.shape[0], 1]), axis=1)
        ints = np.all(dataset.data.astype(int) == dataset.data, axis=0)
        ct = ColumnTransformer([
            ("OneHot", OneHotEncoder(sparse=False, handle_unknown="ignore"), ints),
            ("Normalize", StandardScaler(), ~ints)
        ])
        pf = PolynomialFeatures(3, include_bias=False)
        preprocessing = Pipeline([
            ("Feature Preprocessing", ct),
            ("Polynomial Features 3rd degree", pf)
        ])
        dataset.data = preprocessing.fit_transform(dataset.data)
        train_data, dev_data, train_target, dev_target = train_test_split(dataset.data, dataset.target,
                                                                          test_size=args.dev_size,
                                                                          random_state=args.seed)
        weights = generator.uniform(size=train_data.shape[1])
        train_rmses, dev_rmses = [], []
        for _ in range(args.epochs):
            permutation = generator.permutation(train_data.shape[0])

            for i in range(int(np.ceil(train_data.shape[0] / args.batch_size))):
                batch_data = train_data[permutation[i * args.batch_size:(i + 1) * args.batch_size]]
                batch_target = train_target[permutation[i * args.batch_size:(i + 1) * args.batch_size]]
                batch_prediction = batch_data @ weights
                gradient = (batch_data.T @ (batch_prediction - batch_target) + args.alpha * weights) \
                           / batch_data.shape[0]
                weights = weights - args.learning_rate * gradient

            train_rmse = mean_squared_error(train_data @ weights, train_target, squared=False)
            dev_rmse = mean_squared_error(dev_data @ weights, dev_target, squared=False)
            wandb.log({"train_rmse": train_rmse, "dev_rmse": dev_rmse})
            train_rmses.append(train_rmse)
            dev_rmses.append(dev_rmse)

        l = LinearRegression().fit(train_data, train_target)
        l.coef_ = weights
        model = Pipeline([
            ("Feature Preprocessing", ct),
            ("Polynomial Features 3rd degree", pf),
            ("Lin", l)
        ])
        print(mean_squared_error(model.predict(dev_data), dev_target, squared=False))
        # Serialize the model.
        # with lzma.open(args.model_path, "wb") as model_file:
        #     pickle.dump(best_model, model_file + "ridge_alpha10")

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
