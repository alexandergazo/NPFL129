#!/usr/bin/env python3
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from scipy import sparse
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from itertools import islice

def iter_window(seq, window_size):
    it = iter(seq)
    result = tuple(islice(it, window_size))
    if len(result) == window_size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.target = dataset_file.read().replace('\n', ' ')
        self.data = self.target.translate(self.DIA_TO_NODIA)


class Dummy():
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y):
        pass

    def transform(self, X):
        X = np.vstack([
                np.zeros((self.window_size // 2, X.shape[1])),
                X,
                np.zeros((self.window_size // 2, X.shape[1]))])
        X_out = np.empty((X.shape[0] - self.window_size + 1, X.shape[1] * self.window_size))
        for i, window in enumerate(iter_window(X, self.window_size)):
            X_out[i] = np.concatenate(window)
        X_out = sparse.csr_matrix(X_out)
        return X_out


def generate_data(data, target, window_size=5):
    data_size = data.shape[0] - window_size + 1
    X = np.empty((data_size, data.shape[1] * window_size))
    y = np.empty((data_size, data.shape[1]))
    for i, window in tqdm(enumerate(iter_window(data, window_size)), total=data_size):
        X[i] = np.concatenate(window)
        y[i] = target[i + window_size // 2]
    X = sparse.csr_matrix(X)
    return X, y


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--window_size", default=9, type=int, help="Default window size (odd).")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)

        if os.path.isfile("xXw.npz") and os.path.isfile("y.npz"):
            X = sparse.load_npz("X.npz")
            y = np.load("y.npz")["arr_0"]
        else:
            train = Dataset()
            data = list(train.data.lower())
            target = list(train.target.lower())
            o = OneHotEncoder(dtype=np.bool_, handle_unknown='ignore', sparse=False)
            o.fit(np.reshape(target + data, (-1, 1)))
            data = o.transform(np.reshape(data, (-1, 1)))
            target = o.transform(np.reshape(target, (-1, 1)))
            X, y = generate_data(data, target, args.window_size)
            sparse.save_npz("xX.npz", X)
            np.savez_compressed("yy.npz", y)

        print("Training started.", X.shape)
        mlp = MLPClassifier((500,100), early_stopping=True, verbose=11)
        # mlp = MLPClassifier((300, 100, 100), activation='tanh', early_stopping=True, verbose=11, max_iter=30, alpha=0.001, learning_rate_init=0.0002)
        mlp.fit(X, y)
        model = make_pipeline(o, Dummy(args.window_size), mlp, verbose=11)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        friends = defaultdict(list)
        friends.update({'a': ['á'], 'c': ['č'], 'd': ['ď'], 'e': ['é', 'ě'], 'i': ['í'], 'n': ['ň'], 'o': ['ó'], 'r': ['ř'], 's': ['š'], 't': ['ť'], 'u': ['ú', 'ů'], 'y': ['ý'], 'z': ['ž']})

        test = Dataset(args.predict)
        X_whole = np.reshape(list(test.data.lower()), (-1, 1))

        with lzma.open("diacritization.model", "rb") as model_file:
            model = pickle.load(model_file)
        # mlp = model.named_steps['mlpclassifier']
        # for i in range(len(mlp.coefs_)):
        #     mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        # for i in range(len(mlp.intercepts_)):
        #     mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
        # model = make_pipeline(model.named_steps['onehotencoder'], Dummy(args.window_size), mlp)
        # with lzma.open(args.model_path, "wb") as model_file:
        #     pickle.dump(model, model_file)
        # return

        predictions = []
        batch_size = 5000
        for i in range(X_whole.shape[0] // batch_size + 1):
            X = X_whole[i * batch_size : (i + 1) * batch_size]
            original = test.data[i * batch_size : (i + 1) * batch_size]
            y = model.predict(X)
            y = model.named_steps['onehotencoder'].inverse_transform(y)

            for predicted, orig in zip(y.flatten(), original):
                if predicted not in friends[orig.lower()]:
                    predictions.append(orig)
                else:
                    if orig.isupper():
                        predictions.append(predicted.upper())
                    else:
                        predictions.append(predicted)

        predictions = ''.join(predictions)
        return predictions
        # for dictum, target, _  in zip(X, y, range(20)):
            # print(''.join(o.inverse_transform(np.reshape(dictum, (21, -1))).flatten().tolist()))
            # print(o.inverse_transform(np.reshape(target, (1, -1))))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
