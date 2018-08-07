import numpy as np
from utils import check_data
import pandas as pd
from sklearn.datasets import load_iris


class Naive_Bayes():

    def __init__(self, smooth=1):
        """Initialize NB classifier

           Parameters
           ----------
           distribution : str, {"Gaussian", "Multinomial", "Bernoulli"}
           Distribution used to calculate probability.

           Returns
           -------
           None
           """
        self.smooth = smooth

    def conditional_prob(self):
        pass


class MultinomialNB():

    def __init__(self, smooth=1):
        self.smooth = smooth
        self.cond_prob = None

    def _prior_prob(self, y):
        y_unique, y_counts = np.unique(y, return_counts=True)
        prior = y_counts / len(y)
        return prior

    def _split_data(self, X, y):
        data_split = {}
        y, c = check_data.check_y(y)
        full_data = np.concatenate((X, y), axis=1)
        y_unique = np.unique(y)
        for y_value in y_unique:
            data_split[y_value] = full_data[np.where(full_data[:,-1] == y_value)]
        return data_split

    def _conditional_prob(self, data_split):
        y_unique = list(data_split.keys()) # number of classes
        n = data_split[y_unique[0]].shape[1] # number of features
        print(data_split)
        counts = []
        totals = []
        for y in data_split.keys():
            temp_count = np.sum(data_split[y], axis=0).tolist()
            total = np.sum(data_split[y])
            counts.append(temp_count)
            totals.append(total)
        counts = np.array(counts)
        totals = np.array(totals).reshape(len(y_unique), 1)
        self.cond_prob = (counts + self.smooth) / (totals + n)
        print(counts)
        print(totals)

    def fit(self, X, y):
        X = check_data.check_X(X)
        y, c = check_data.check_y(y)
        splitted_data = self._split_data(X, y)
        self._conditional_prob(splitted_data)

    def predict(self, X):
        pass


if __name__ == "__main__":
    # iris = load_iris()
    # X, y = iris.data, iris.target

    nb = MultinomialNB()

    # Multinomial Toy Data
    X = np.array([[2, 1, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0],
         [1, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 1, 1]])

    y = np.array([1, 1, 1, 0])

    X_test = np.array([[3, 0, 0, 0, 1, 1]])
    X_y = np.array([1])

    nb.fit(X, y)
    print(nb.cond_prob)


