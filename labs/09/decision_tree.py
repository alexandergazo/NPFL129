#!/usr/bin/env python3
import argparse
import pandas as pd

from collections import deque
from scipy.stats import mode
import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset:
    """
    This class is a representation of a (subset of) dataset optimized for splitting needed for construction of
    regression trees. Actual data are not copied, indices are kept, only.
    """

    def __init__(self, df, ix=None):
        """
        Constructor
        :param df: Pandas DataFrame or another Dataset instance. In the latter case only meta data are copied.
        :param ix: boolean index describing samples selected from the original dataset
        """
        if isinstance(df, pd.DataFrame):
            self.columns = list(df.columns)
            self.cdict = {c: i for i, c in enumerate(df.columns)}
            self.data = [df[c].values for c in self.columns]
        elif isinstance(df, Dataset):
            self.columns = df.columns
            self.cdict = df.cdict
            self.data = df.data
            assert ix is not None
        self.ix = np.arange(len(self.data[0]), dtype=np.int64) if ix is None else ix

    def __getitem__(self, cname):
        """
        Returns dataset column.
        :param cname: column name
        :return: the column as a numpy array
        """
        return self.data[self.cdict[cname]][self.ix]

    def __len__(self):
        """
        The number of samples
        :return:
        """
        return len(self.ix)

    def to_dict(self):
        """
        Return the data in a form used in prediction.
        :return: list of dicts with dict for each data sample, keys are the column names
        """
        return [{c: self.data[self.cdict[c]][i] for c in self.columns} for i in self.ix]

    def modify_col(self, cname, d):
        """
        Creates a copy of this dataset replacing one of its columns data. This method might be helpful for the Gradient
        Boosted Trees.
        :param cname: column name
        :param d: a numpy array with new column data
        :return: new Dataset
        """
        assert len(self.ix) == len(self.data[0]), 'works for unfiltered rows, only'
        new_dataset = Dataset(self, ix=self.ix)
        new_dataset.data = list(self.data)
        new_dataset.data[self.cdict[cname]] = d
        return new_dataset

    def filter_rows(self, cname, cond):
        """
        Creates a new Dataset containing only the rows satisfying a given condition.
        :param cname: column name
        :param cond: condition
        :return:
        """
        col = self[cname]
        return Dataset(self, ix=self.ix[cond(col)])

    def get_bootstrap(self, rng=np.random.RandomState(1)):
        bootstrap_indices = rng.choice(a=range(len(self.ix)), size=len(self.ix))
        new_dataset = Dataset(self, ix=self.ix)
        new_dataset.data = list(self.data)
        new_dataset.data = [self.data[i][bootstrap_indices] for i in range(len(self.columns))]
        return new_dataset


def gini_criterion(data, tattr):
    probs = np.unique(data[tattr], return_counts=True)[1] / len(data)
    return len(data) * (probs * (1 - probs)).sum()


def entropy_criterion(data, tattr):
    probs = np.unique(data[tattr], return_counts=True)[1] / len(data)
    entropy = -(probs * np.log(probs)).sum()
    return len(data) * entropy


def get_criterion(name):
    if name == 'gini':
        return gini_criterion
    elif name == 'entropy':
        return entropy_criterion
    else:
        raise ValueError(f'Unsupported criterion "{name}"')


class DecisionNode:
    def __init__(self, attr, value, left, right):
        """
        Constructs a node
        :param attr: splitting attribute
        :param value: splitting attribute value
        :param left: left child
        :param right: right child
        """
        self.attr = attr
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        """
        Evaluates the node.
        :param x: a dictionary: key = attribute (column) name, value = attribute value
        :return: reference to the corresponding child
        """
        if isinstance(self.value, str):
            if x[self.attr] == self.value:
                return self.left.evaluate(x)
            else:
                return self.right.evaluate(x)
        else:
            if x[self.attr] <= self.value:
                return self.left.evaluate(x)
            else:
                return self.right.evaluate(x)

    def get_nodes(self):
        """
        Return all nodes of the subtree rooted in this node.
        """
        ns = []
        q = deque([self])
        while len(q) > 0:
            n = q.popleft()
            ns.append(n)
            if isinstance(n, DecisionNode):
                q.append(n.left)
                q.append(n.right)
        return ns

    def __str__(self):
        """
        String representation f
        :return:
        """
        if isinstance(self.value, str):
            return '{} == "{}"'.format(self.attr, self.value)
        else:
            return '{} <={:5.2f}'.format(self.attr, self.value)


class LeafNode:
    def __init__(self, response):
        self.response = response

    def evaluate(self, x):
        return self.response

    def get_nodes(self):
        return [self]

    def __str__(self):
        return '{:5.2f}'.format(self.response)


def evaluate_all(model, data):
    """
    Makes predictions for all dataset samples.
    :param model: any model implementing evaluate(x) method
    :param data: Dataset instance
    :return: predictions as a numpy array
    """
    return np.r_[[model.evaluate(x) for x in data.to_dict()]]


class ClassificationTree:
    def __init__(self, data, tattr, xattrs=None, criterion='gini', max_depth=5,
                 feature_subsampling=1,
                 min_to_split=2,
                 rng=np.random.default_rng(1)):
        """
        Regression tree constructor. Constructs the model fitting supplied dataset.
        :param data: Dataset instance
        :param criterion: Callable of the criterion
        :param tattr: the name of target attribute column
        :param xattrs: list of names of the input attribute columns
        :param max_depth: limit on tree depth
        :param feature_subsampling: float fraction of used features
        :param rng: random number generator used for sampling features when selecting a split candidate
        """
        self.xattrs = [c for c in data.columns if c != tattr] if xattrs is None else np.array(xattrs)
        if isinstance(criterion, str):
            self.criterion = get_criterion(criterion)
        elif isinstance(criterion, callable):
            self.criterion = criterion
        else:
            raise ValueError('Criterion must be string or callable.')
        self.tattr = tattr
        self.feature_subsampling = feature_subsampling
        self.rng = rng
        self.min_to_split = min_to_split
        self.root = self.build_tree(data, self.criterion(data, tattr),
                                    max_depth=np.inf if max_depth is None else max_depth)

    def evaluate(self, x):
        """
        Evaluates the node.
        :param x: a dictionary: key = attribute (column) name, value = attribute value
        :return: reference to the corresponding child
        """
        return self.root.evaluate(x)

    def build_tree(self, data, criterion_value, max_depth):
        if max_depth > 0 and len(data) >= self.min_to_split and criterion_value:
            best_crit = criterion_value
            xattrs = self.xattrs[self.rng.uniform(size=len(self.xattrs)) <= self.feature_subsampling]
            for xattr in xattrs:
                x = np.sort(data[xattr])
                if not np.issubdtype(x.dtype, np.number):
                    continue
                split_points = (x[1:] + x[:-1]) / 2
                if len(split_points) <= 1: continue
                for split_point in split_points:
                    data_l = data.filter_rows(xattr, lambda a: a <= split_point)
                    data_r = data.filter_rows(xattr, lambda a: a > split_point)

                    crit_l = self.criterion(data_l, self.tattr)
                    crit_r = self.criterion(data_r, self.tattr)

                    total_crit = crit_l + crit_r

                    if total_crit < best_crit and len(data_l) > 0 and len(data_r) > 0:
                        best_crit, best_xattr, best_split = total_crit, xattr, split_point
                        best_data_l, best_data_r = data_l, data_r
                        best_crit_l, best_crit_r = crit_l, crit_r

            if best_crit < criterion_value:
                return DecisionNode(best_xattr, best_split,
                                    self.build_tree(best_data_l, best_crit_l, max_depth - 1),
                                    self.build_tree(best_data_r, best_crit_r, max_depth - 1))
        return LeafNode(mode(data[self.tattr]).mode[0])

    def plot(self):
        """
        Plots trees. Useful for debugging. You have to install networkx and pydot Python modules as well as graphviz.
        Display in Jupyter notebook or save the plot to file:
            img = tree.plot()
            with open('tree.png', 'wb') as f:
            f.write(img.data)
        """
        import networkx as nx
        g = nx.DiGraph()
        V = self.root.get_nodes()
        d = {}
        for i, n in enumerate(V):
            d[n] = i
            g.add_node(i, label='{}'.format(n))
        for n in V:
            if isinstance(n, DecisionNode):
                g.add_edge(d[n], d[n.left])
                g.add_edge(d[n], d[n.right])

        dot = nx.drawing.nx_pydot.to_pydot(g)
        dot.write_png("test.png")
        return dot.create_png()

def main(args):
    # Use the wine dataset
    dataset = sklearn.datasets.load_wine(as_frame=True)

    train_data, test_data = sklearn.model_selection.train_test_split(
        dataset.frame, test_size=args.test_size, random_state=args.seed)

    # TODO
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    train_data = Dataset(train_data)
    test_data = Dataset(test_data)
    classificator = ClassificationTree(train_data, xattrs=dataset.feature_names, tattr='target',
                                       min_to_split=args.min_to_split, max_depth=args.max_depth,
                                       criterion=args.criterion)

    if not args.recodex: classificator.plot()

    train_predictions = evaluate_all(classificator, train_data)
    test_predictions = evaluate_all(classificator, test_data)

    train_accuracy = accuracy_score(train_data['target'], train_predictions)
    test_accuracy = accuracy_score(test_data['target'], test_predictions)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
