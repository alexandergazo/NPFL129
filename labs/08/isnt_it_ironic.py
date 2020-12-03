#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.zip",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with zipfile.ZipFile(name, "r") as dataset_file:
            with dataset_file.open(os.path.basename(name).replace(".zip", ".txt"), "r") as train_file:
                for line in train_file:
                    label, text = line.decode("utf-8").rstrip("\n").split("\t")
                    self.data.append(text)
                    self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")
parser.add_argument("--test_size", default=1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args):
    if True:
        # We are training a model.
        rng = np.random.default_rng(args.seed)
        train = Dataset()

        train_data, dev_data, train_target, dev_target = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=args.test_size, random_state=args.seed)

        tfidf_chars = TfidfVectorizer(decode_error='ignore', strip_accents='ascii',
                                      analyzer='char_wb', ngram_range=(1,3), max_features=1000)
        tfidf_words = TfidfVectorizer(decode_error='ignore', strip_accents='ascii',
                                      lowercase=False, ngram_range=(1,4), max_features=1000)
        model = MLPClassifier((800, 300), batch_size=500, learning_rate_init=1e-4, alpha=1e-5, verbose=1)

        char_features = tfidf_chars.fit_transform(train_data)
        word_features = tfidf_words.fit_transform(train_data)
        features = np.append(char_features.toarray(), word_features.toarray(), 1)

        episodes = 25
        model.max_iter = 1
        model.fit(features, train_target)
        try:
            for i in range(episodes):
                model.partial_fit(features, train_target)
                char_features = tfidf_chars.transform(dev_data)
                word_features = tfidf_words.transform(dev_data)
                dev_features = np.append(char_features.toarray(), word_features.toarray(), 1)
                print("F1 Score: {:.2f}%".format(100 * f1_score(dev_target, model.predict(dev_features))))
        except KeyboardInterrupt:
            print("Training interrupted.")

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

        with lzma.open(args.model_path + ".tfidf_char", "wb") as model_file:
            pickle.dump(tfidf_chars, model_file)

        with lzma.open(args.model_path + ".tfidf_word", "wb") as model_file:
            pickle.dump(tfidf_words, model_file)

    if True:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        with lzma.open(args.model_path + ".tfidf_char", "rb") as model_file:
            tfidf_chars = pickle.load(model_file)

        with lzma.open(args.model_path + ".tfidf_word", "rb") as model_file:
            tfidf_words = pickle.load(model_file)

        char_features = tfidf_chars.transform(test.data)
        word_features = tfidf_words.transform(test.data)
        features = np.append(char_features.toarray(), word_features.toarray(), 1)

        predictions = model.predict(features)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
