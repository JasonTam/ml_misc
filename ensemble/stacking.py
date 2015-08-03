from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import KFold, StratifiedKFold

from collections import defaultdict

def param_map(params):
    """
    :param params: keyed by a string in the following format:
        #_paramname
        # is the index of the base estimator or 'meta'
    :return:
    """
    split_d = defaultdict(lambda: {})
    for k, v in params.items():
        ind, param_name = k.split('_', 1)
        split_d[ind][param_name] = v
    return split_d


class Stacking(BaseEstimator, ClassifierMixin):
    """Stacked generalization
    Currently for regression
    """

    def __init__(self, base_estimators, meta_estimator,
                 cv=3, retrain=True, one_fold=False):
        """
        :param base_estimators: base level 0 estimators
        :param meta_estimator: meta level 1 estimator
        :param cv: # of folds
        :param retrain: retrain base estimators on full set after meta is trained
        :param one_fold: only use 1 fold split
        :return:
        """
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.cv = cv

        self.retrain = retrain
        self.one_fold = one_fold

    def fit(self, X, y, **fit_params):
        # Parse params
        param_d = param_map(fit_params)

        holdout_base_preds = []
        holdout_ys = []
        kf = KFold(n=len(y), n_folds=self.cv)
        for train_ind, holdout_ind in kf:
            # train_ind, holdout_ind = iter(kf).next()
            X_train, X_holdout = X[train_ind], X[holdout_ind]
            y_train, y_holdout = y[train_ind], y[holdout_ind]

            # Train base estimators
            for ii, est in enumerate(self.base_estimators):
                est.fit(X_train, y_train)

            # Predict hold out set with base estimators
            holdout_base_pred = np.array([est.predict(X_holdout) for est in self.base_estimators]).T
            holdout_base_preds.append(holdout_base_pred)
            holdout_ys.append(y_holdout)

            if self.one_fold:
                break

        X_base_preds = np.concatenate(holdout_base_preds)
        y_base_preds = np.concatenate(holdout_ys)

        # Train meta estimator
        self.meta_estimator.fit(X_base_preds, y_base_preds)

        # Retrain the base estimators on entire set
        if self.retrain:
            for est in self.base_estimators:
                est.fit(X, y)

        return self

    def predict(self, X, **pred_params):
        base_preds = np.array([est.predict(X) for est in self.base_estimators]).T
        final_pred = self.meta_estimator.predict(base_preds)
        return final_pred


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import ExtraTreesRegressor

    from sklearn.datasets import load_boston
    boston = load_boston()
    X, y = boston.data, boston.target

    base_ests = [SVR(),
                 KNeighborsRegressor(n_neighbors=20),
                 ExtraTreesRegressor(n_estimators=400),
                 ]
    meta_est = LinearRegression()

    scores_baseline = []
    scores = []
    kf = KFold(len(y), n_folds=4)
    for train_ind, val_ind in kf:
        X_train, X_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]
        stack = Stacking(base_ests, meta_est, cv=4)

        stack.fit(X_train, y_train)

        y_pred = stack.predict(X_val)
        score = mse(y_val, y_pred)
        print score
        scores.append(score)

        q = ExtraTreesRegressor(n_estimators=400)
        q.fit(X_train, y_train)
        scores_baseline.append(mse(y_val, q.predict(X_val)))

    print
    print 'Scores \t\t\t', np.mean(scores)
    print 'ET baseline \t', np.mean(scores_baseline)