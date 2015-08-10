from rpforest import RPForest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone



class rpfnn(BaseEstimator, ClassifierMixin):
    def __init__(self, leaf_size=50, no_trees=10):
        self.model = RPForest(leaf_size, no_trees)
        self.y = None

    def fit(self, X, y, **fit_params):
        self.model.fit(X)
        self.y = y

    def predict(self, X, number=10, **pred_params):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X[:, None]

        ret = np.array([self.predict1(X_i, number) for X_i in X])
        return ret

    def predict1(self, X, number):
        q = self.model.query(X, number)
        y_q = [self.y[q_i] for q_i in q]

        return np.mean(y_q)


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.svm import SVR, SVC
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.lda import LDA
    from sklearn.cross_validation import KFold

    from sklearn.datasets import load_boston
    boston = load_boston()
    X, y = boston.data, boston.target
    _, bins = np.histogram(y, bins=10)
    y_binned = np.digitize(y, bins=bins)

    scores_baseline = []
    scores = []
    kf = KFold(len(y), n_folds=4)
    for train_ind, val_ind in kf:
        X_train, X_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]

        clf = rpfnn()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        score = mse(y_val, y_pred)
        print score
        scores.append(score)

        # q = ExtraTreesRegressor(n_estimators=400)
        q = LinearRegression()
        q.fit(X_train, y_train)
        scores_baseline.append(mse(y_val, q.predict(X_val)))

    print
    print 'Scores \t\t\t', np.mean(scores)
    print 'ET baseline \t', np.mean(scores_baseline)

