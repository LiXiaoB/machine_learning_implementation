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
        self.prior = None
        self.cond_prob = None

    def _prior_prob(self, y):
        y_unique, y_counts = np.unique(y, return_counts=True)
        self.prior = y_counts / len(y)

    def _split_data(self, X, y):
        X_splitted = {}
        full_data = np.concatenate((X, y), axis=1)
        y_unique = np.unique(y)
        for y_value in y_unique:
            data_temp = full_data[np.where(full_data[:,-1] == y_value)]
            X_temp = np.delete(data_temp, -1, axis=1)
            X_splitted[y_value] = X_temp
        return X_splitted

    def _conditional_prob(self, data_split):
        y_unique = list(data_split.keys()) # number of classes
        n = data_split[y_unique[0]].shape[1] # number of features
        # print(data_split)
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
        # print(counts)
        # print(totals)

    def fit(self, X, y):
        X = check_data.check_X(X)
        y, c = check_data.check_y(y)
        self._prior_prob(y)
        splitted_data = self._split_data(X, y)
        self._conditional_prob(splitted_data)

    def predict(self, X):
        prior = self.prior
        cond_prob = self.cond_prob
        P_class = np.ones(shape=(len(X), len(prior)))
        for i in range(len(X)):
            prob = np.prod(np.power(cond_prob, X[i]), axis=1)
            prob = np.multiply(prior, prob)
            P_class[i] = prob
        prediction = np.argmax(P_class, axis=1)
        return prediction


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

    X_test = np.array([[3, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1, 1]])
    y_test = np.array([[1], [0]])

    # run
    nb.fit(X, y)
    pred = nb.predict(X_test)
    print(pred)

