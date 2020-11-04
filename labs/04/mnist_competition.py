#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter, rotate, affine_transform, map_coordinates
from sklearn.neural_network import MLPClassifier


class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """

    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)
        self.data = self.data.reshape([-1, 28 * 28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


def get_shift(size, sigma, alpha):
    pos = np.random.uniform(size=size, low=-1, high=1)
    pos = gaussian_filter(pos, sigma)
    pos = pos / np.linalg.norm(pos) * alpha
    return pos.round().astype('int')


def transform_instance(img, predict, sigma=6, alpha=38, beta=15, gamma=20, cval=-1):
    shift_x = get_shift((28, 28), sigma, alpha) + np.asarray([[i] * 28 for i in range(28)])
    shift_y = get_shift((28, 28), sigma, alpha) + np.asarray([[i] * 28 for i in range(28)]).T
    new_img = map_coordinates(img.reshape(28, 28), (shift_x, shift_y), order=2, cval=cval)

    if predict == 1 or predict == 7:
        beta /= 2
    new_img = rotate(new_img, np.random.uniform(low=-beta, high=beta), cval=cval, output=np.array(img.reshape(28, 28)),
                     order=1, reshape=False)

    new_img = affine_transform(new_img, np.random.uniform(low=1 - gamma / 100, high=1 + gamma / 100) * np.eye(2),
                               output_shape=(28, 28), cval=cval, order=1)
    return new_img.flatten()


def main(args):
    if args.predict is None:
        from tqdm import tqdm
        np.random.seed(args.seed)
        train = Dataset()

        train.data = train.data / 255 * 2 - 1
        train_data, test_data, train_target, test_target = train_test_split(
            train.data, train.target, stratify=train.target, test_size=0.1, random_state=args.seed)
        iters = 20
        train_data_expanded = np.empty((train_data.shape[0] * iters, train_data.shape[1]))
        train_target_expanded = np.empty((train_target.shape[0] * iters,))
        for idx, instance in enumerate(tqdm(train_data)):
            for i in range(iters):
                train_data_expanded[idx * iters + i, :] = transform_instance(instance, train_target[idx])
                train_target_expanded[idx * iters + i] = train_target[idx]
        np.save("train_data_expanded", train_data_expanded)
        np.save("train_target_expanded", train_target_expanded)

        mlp = MLPClassifier((1000, 500), random_state=args.seed, verbose=10)
        # mlp = MLPClassifier((1000, 500), random_state=args.seed, verbose=10, learning_rate_init=0.1, solver='sgd',
        #                     activation='tanh', learning_rate='adaptive', momentum=0, max_iter=iters)
        mlp.fit(train_data_expanded, train_target_expanded)
        print(accuracy_score(test_target, mlp.predict(test_data)))

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(mlp, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        predictions = model.predict(test.data)
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
