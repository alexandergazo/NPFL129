#!/usr/bin/env python3
import argparse

from decision_tree import ClassificationTree, Dataset, evaluate_all

import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score
import sklearn.model_selection
from scipy.stats import mode

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrapping", default=False, action="store_true", help="Perform data bootstrapping")
parser.add_argument("--feature_subsampling", default=1, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the wine dataset
    dataset = sklearn.datasets.load_wine(as_frame=True)

    train_data, test_data = sklearn.model_selection.train_test_split(
        dataset.frame, test_size=args.test_size, random_state=args.seed)

    rng = np.random.RandomState(args.seed)

    train_data, test_data = Dataset(train_data), Dataset(test_data)

    forest = [ClassificationTree(train_data.get_bootstrap(rng=rng) if args.bootstrapping else train_data,
                                 xattrs=dataset.feature_names, tattr='target', rng=rng,
                                 criterion='entropy', max_depth=args.max_depth,
                                 feature_subsampling=args.feature_subsampling)
              for _ in range(args.trees)]

    # haskell? pls
    test_predictions = mode(np.array(list(map(lambda tree: evaluate_all(tree, test_data), forest))), axis=0).mode.flatten()
    train_predictions = mode(np.array(list(map(lambda tree: evaluate_all(tree, train_data), forest))), axis=0).mode.flatten()

    train_accuracy = accuracy_score(train_data['target'], train_predictions)
    test_accuracy = accuracy_score(test_data['target'], test_predictions)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
