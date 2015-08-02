from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import KFold, StratifiedKFold


class Stacking(BaseEstimator, ClassifierMixin):
    """Stacked generalization
    Currently for regression
    """

    def __init__(self, base_estimators, meta_estimator, cv=3, retrain=True):
        """
        """
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.cv = cv

        self.retrain = retrain

    def fit(self, X, y, **fit_params):
        kf = KFold(n=len(y), n_folds=self.cv)
        # for train_ind, holdout_ind in kf:
        train_ind, holdout_ind = iter(kf).next()
        X_train, X_holdout = X[train_ind], X[holdout_ind]
        y_train, y_holdout = y[train_ind], y[holdout_ind]

        # Train base estimators
        for est in self.base_estimators:
            est.fit(X_train, y_train)

        # Predict hold out set with base estimators
        holdout_base_preds = np.array([est.predict(X_holdout) for est in self.base_estimators]).T

        # Train meta estimator
        self.meta_estimator.fit(holdout_base_preds, y_holdout)

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
    from sklearn.ensemble import ExtraTreesRegressor

    from sklearn.datasets import load_boston
    boston = load_boston()
    X, y = boston.data, boston.target

    base_ests = [LinearRegression(), Ridge(), Lasso()]
    meta_est = ExtraTreesRegressor(n_estimators=200)


    scores_baseline = []
    scores = []
    kf = KFold(len(y), n_folds=4)
    for train_ind, val_ind in kf:
        X_train, X_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]
        stack = Stacking(base_ests, meta_est, cv=3)

        stack.fit(X_train, y_train)

        y_pred = stack.predict(X_val)
        score = mse(y_val, y_pred)
        print score
        scores.append(score)

        q = LinearRegression()
        q.fit(X_train, y_train)
        scores_baseline.append(mse(y_val, q.predict(X_val)))

print
print np.mean(scores)
print np.mean(scores_baseline)