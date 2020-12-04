#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    if args.naive_bayes_type == 'gaussian':
        means = np.empty((args.classes, data.shape[1]))
        variances = np.empty_like(means)
        class_prob = np.empty(args.classes)
        for i in range(args.classes):
            means[i] = np.mean(train_data[train_target == i], axis=0)
            variances[i] = np.var(train_data[train_target == i], axis=0) + args.alpha
            class_prob[i] = (train_target == i).sum() / train_data.shape[0]

        from scipy.stats import norm
        predictions = []
        for dictum in test_data:
            probs = [class_prob[i] * np.prod(norm.pdf(dictum, means[i], np.sqrt(variances[i])))
                     for i in range(args.classes)]
            predictions.append(np.argmax(probs))
    elif args.naive_bayes_type == 'multinomial':
        biases = np.array([(train_target == i).sum() for i in range(args.classes)]) / train_data.shape[0]
        weights = np.empty((args.classes, data.shape[1]))
        for i in range(args.classes):
            sums = np.sum(train_data[train_target == i], axis=0)
            weights[i] = (sums + args.alpha) / (sums.sum() + args.alpha * data.shape[1])

        weights, biases = np.log(weights), np.log(biases)

        predictions = []
        for dictum in test_data:
            probs = weights @ dictum + biases
            predictions.append(np.argmax(probs))

    elif args.naive_bayes_type == 'bernoulli':
        biases = np.array([(train_target == i).sum() for i in range(args.classes)]) / train_data.shape[0]
        p = np.empty((args.classes, data.shape[1]))
        for i in range(args.classes):
            sums = np.sum(train_data[train_target == i] > 0, axis=0)
            p[i] = (sums + args.alpha) / ((train_target == i).sum() + args.alpha * 2)

        biases = np.log(biases)

        predictions = []
        for dictum in test_data:
            probs = np.log(p) @ (dictum > 0) + np.log(1 - p) @ (dictum == 0) + biases
            predictions.append(np.argmax(probs))
    else:
        raise ValueError(f'Unknown naive_bayes_type value "{args.naive_bayes_type}".')

    test_accuracy = accuracy_score(test_target, predictions)
    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))
