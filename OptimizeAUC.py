import numpy as np
from functools import partial
from scipy.optimize import fmin
from sklearn import metrics


class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        # multiply element k of coef with column k
        x_coef = X * coef

        # create preds by taking row wise sum
        predictions = np.sum(x_coef, axis=1)

        # calculate aux score
        auc_score = metrics.roc_auc_score(y, predictions)

        # return negative auc cause we work with a minimization function
        return -1.0 * auc_score

    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)

        # initialization for best parameter search
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)

        # minimization function (of negative auc)
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        # like in auc function
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions
