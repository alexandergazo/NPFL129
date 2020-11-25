#!/usr/bin/env python3
from diacritization_eval import accuracy
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
    def __init__(self, window_size, ngram):
        self.window_size = window_size
        self.ngram = ngram

    def fit(self, X, y):
        pass

    def transform(self, X):
        X = [''] * (self.window_size // 2) + X + [''] * (self.window_size // 2)
        X_out = np.empty((len(X) - self.window_size + 1,
                          self.window_size + (self.window_size - self.ngram + 1)),
                         dtype=f'<U{self.ngram}')
        for i, window in enumerate(iter_window(X, self.window_size)):
            X_out[i] = list(window) + [''.join(i) for i in iter_window(window, self.ngram)]
        return X_out


def generate_data(data, target, window_size, ngram):
    data_size = len(data) - window_size + 1
    X = np.empty((data_size, window_size + (window_size - ngram + 1)), dtype=f'<U{ngram}')
    y = np.empty(data_size)
    for i, window in tqdm(enumerate(iter_window(data, window_size)), total=data_size):
        X[i] = list(window) + [''.join(i) for i in iter_window(window, ngram)]
        y[i] = target[i + window_size // 2]
    return X, y


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="lastlast.model", type=str, help="Model path")
parser.add_argument("--window_size", default=9, type=int, help="Default window size (odd).")
parser.add_argument("--train", default=None, type=str, help="Train on given data")
parser.add_argument("--ngram", default=5, type=int, help="Default window size (odd).")


def mark(letter):
    if letter in 'áéíúóý':
        return 1
    elif letter in 'čďěňřšťž':
        return 2
    elif letter == 'ů':
        return 3
    else:
        return 0

def demark(letter, mark):
    if mark == 0:
        return letter
    if mark == 1 and letter in 'aeiuoy':
        return 'áéíúóý'['aeiuoy'.index(letter)]
    if mark == 2 and letter in 'cdenrstz':
        return 'čďěňřšťž'['cdenrstz'.index(letter)]
    if mark == 3 and letter == 'u':
        return 'ů'
    return letter


def main(args):
    if args.train is not None:
        # We are training a model.
        np.random.seed(args.seed)

        train = Dataset(args.train)
        data = list(train.data.lower())
        target = list(train.target.lower())
        target = [mark(letter.lower()) for letter in target]
        X, y = generate_data(data, target, args.window_size, args.ngram)
        o = OneHotEncoder(handle_unknown='ignore')
        X = o.fit_transform(X)

        print("Training started.", X.shape)
        lr = LogisticRegression(class_weight='balanced', C=1000, verbose=1, n_jobs=1, max_iter=10000).fit(X, y)
        model = make_pipeline(Dummy(args.window_size, args.ngram), o, lr, verbose=11)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    if args.predict is not None:
        friends = defaultdict(list)
        friends.update({'a': ['á'], 'c': ['č'], 'd': ['ď'], 'e': ['é', 'ě'], 'i': ['í'], 'n': ['ň'], 'o': ['ó'], 'r': ['ř'], 's': ['š'], 't': ['ť'], 'u': ['ú', 'ů'], 'y': ['ý'], 'z': ['ž']})

        test = Dataset(args.predict)
        X_whole = list(test.data.lower())

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        from diacritization_dictionary import Dictionary
        from itertools import chain
        variants = Dictionary().variants
        predictions = []
        batch_size = 5000
        word = ""
        for i in range(len(X_whole) // batch_size + 1):
            X = X_whole[i * batch_size : (i + 1) * batch_size]
            original = test.data[i * batch_size : (i + 1) * batch_size]
            y = model.predict(X)
            y = np.reshape([demark(l, m) for l, m in zip(X, y)], (-1, 1))

            for predicted, orig in zip(y.flatten(), original):
                if orig == ' ':
                    nodia = word.translate(test.DIA_TO_NODIA)
                    if nodia not in variants:
                        val = word
                    else:
                        v = variants[nodia]
                        w = np.asarray(list(map(list, v)))
                        idx = (w != np.asarray(list(word))).sum(axis=1).argmin()
                        val = v[idx]
                    predictions.append(val)
                    word = ""
                    continue
                if predicted not in friends[orig.lower()]:
                    word += orig
                else:
                    if orig.isupper():
                        word += predicted.upper()
                    else:
                        word += predicted
        predictions.append(word)

        predictions = ' '.join(predictions)
        print(predictions)
        if test.target:
            print(f"The accuracy on validation set: {accuracy(test.target, predictions)}")
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
