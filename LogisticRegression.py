from sklearn.linear_model import  LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score , log_loss
import numpy as np
dataset =datasets.load_breast_cancer()
X,y = dataset.data, dataset.target
print(X.shape, y.shape)
print(X[1])

def sigmoid(x):
  return 1/(1 + np.exp(-x))


def sigmoid(x):
  return 1/(1 + np.exp(-x))

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegressionCustom:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            y_predicts = sigmoid(z)

            loss = log_loss(y, y_predicts)
            print("Iteration", _, "Log Loss:", loss)

            dw = (1/n_samples) * np.dot(X.T, y_predicts - y)
            db = (1/n_samples) * np.sum(y_predicts - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_predicts = sigmoid(z)
        return y_predicts

    def predict(self, X, threshold=0.5):
        y_prob = self.predict_proba(X)
        y_pred = [1 if p >= threshold else 0 for p in y_prob]
        return y_pred

# Your code for loading data remains the same

clf = LogisticRegressionCustom()
clf.fit(X, y)

y_predicts = clf.predict(X)
print(y_predicts)

acc = accuracy_score(y, y_predicts)
print("Accuracy:", acc)
