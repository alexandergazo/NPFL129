#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

# TODO if this was vectorized it would be great
def kernel(args, x, y):
    if args.kernel == 'rbf':
        return np.exp(-args.kernel_gamma * np.linalg.norm(x - y) ** 2)
    elif args.kernel == 'poly':
        d, r = args.kernel_gamma * np.dot(x, y) + 1, 1
        for _ in range(args.kernel_degree): r *= d
        return r


# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(args, K, train_data, train_target, test_data=None, test_target=None):
    def predict(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.zeros(X.shape[0])
        weights = a * train_target
        for i, w in enumerate(weights):
            if w == 0: continue
            for j, row in enumerate(X):
                y[j] += w * kernel(args, row, train_data[i])
        return y + b
    def predict_train_data(indices):
        y = (a * train_target * K[indices]).sum(axis=1)
        return y + b
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)

            Ei = predict_train_data([i])[0] - train_target[i]
            KKT = (a[i] >= args.C - args.tolerance or train_target[i] * Ei >= -args.tolerance) \
                and (a[i] <= args.tolerance or train_target[i] * Ei <= args.tolerance)
            if not KKT:
                second_derivative_aj = 2 * K[i, j] - K[i, i] - K[j, j]
                if second_derivative_aj > -args.tolerance:
                    continue
                Ej = predict_train_data([j])[0] - train_target[j]
                aj_new = a[j] - train_target[j] * (Ei - Ej) / second_derivative_aj
                equal_targets = train_target[i] == train_target[j]
                L = max(0, a[j] - (args.C - a[i] if equal_targets else a[i]))
                H = min(args.C, a[j] + (a[i] if equal_targets else args.C - a[i]))
                aj_new = np.clip(aj_new, L, H)

                if abs(aj_new - a[j]) < args.tolerance:
                    continue

                ai_new = a[i] - train_target[i] * train_target[j] * (aj_new - a[j])

                bi = b - Ei - train_target[i] * (ai_new - a[i]) * K[i, i] \
                        - train_target[j] * (aj_new - a[j]) * K[j, i]

                bj = b - Ej - train_target[i] * (ai_new - a[i]) * K[i, j] \
                        - train_target[j] * (aj_new - a[j]) * K[j, j]

                if args.tolerance < ai_new < args.C - args.tolerance:
                    b = bi
                elif args.tolerance < aj_new < args.C - args.tolerance:
                    b = bj
                else:
                    b = (bi + bj) / 2
                a[i] = ai_new
                a[j] = aj_new

                as_changed = True

        train_acc = accuracy_score((train_target + 1) / 2, predict_train_data(range(train_data.shape[0])) > 0)
        train_accs.append(train_acc)
        if test_data is not None:
            test_acc = accuracy_score((test_target + 1) / 2, predict(test_data) > 0)
            test_accs.append(test_acc)

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1]), end='')
            if test_data is not None:
                print(", test acc {:.1f}%".format(100 * test_accs[-1]), end='')
            print()

    print("Training finished after iteration {}, train acc {:.1f}%".format(
        len(train_accs), 100 * train_accs[-1]), end='')
    if test_data is not None:
        print(", test acc {:.1f}%".format(100 * test_accs[-1]), end='')
    print()

    support_vectors, support_vector_weights = [], []
    for a, vector, target in zip(a, train_data, train_target):
        if a > args.tolerance:
            support_vectors.append(vector)
            support_vector_weights.append(a * target)

    return support_vectors, support_vector_weights, b, train_accs, test_accs

def main(args):
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    print("Caching kernel...", end="")
    K = np.asarray([[kernel(args, train_data[i], train_data[j])
                     for i in range(train_data.shape[0])]
                    for j in range(train_data.shape[0])])
    print("Done.")
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, K, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:][0], support_vectors[:][1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        predict_function = lambda x: sum(weight * kernel(args, vector, x) for vector, weight in zip(support_vectors, support_vector_weights)) + bias

        plot(predict_function, support_vectors)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
