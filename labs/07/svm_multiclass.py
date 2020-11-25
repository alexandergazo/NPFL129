#!/usr/bin/env python3
import argparse
from collections import namedtuple

import numpy as np
import sklearn.datasets
import sklearn.model_selection
from smo_algorithm import smo, kernel

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    SVM = namedtuple('SVM', 'support_vectors, support_vector_weights, bias, class_list')
    def predict(svm, x):
        return svm.class_list[
            sum(w * kernel(args, x, v) for w, v in zip(svm.support_vector_weights, svm.support_vectors))
            + svm.bias > 0]
    svms = []
    for i in range(args.classes - 1):
        for j in range(i + 1, args.classes):
            i_indices = train_target == i
            j_indices = train_target == j
            target = np.zeros_like(train_target)
            target[i_indices] = 1
            target[j_indices] = -1
            i_test_indices = test_target == i
            j_test_indices = test_target == j
            t_target = np.zeros_like(test_target)
            t_target[i_test_indices] = 1
            t_target[j_test_indices] = -1
            print(f'Training classes {i} and {j}.')
            svm = SVM(
                *smo(args, train_data[i_indices | j_indices], target[i_indices | j_indices], test_data[i_test_indices | j_test_indices], t_target[i_test_indices | j_test_indices])[:3],
                [j, i])
            svms.append(svm)
    test_accuracy = 0
    for dictum, target in zip(test_data, test_target):
        classes = np.zeros(args.classes)
        for svm in svms:
            classes[predict(svm, dictum)] += 1
        test_accuracy += classes.argmax() == target
    test_accuracy /= test_target.shape[0]
    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
