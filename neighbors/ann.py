from annoy import AnnoyIndex
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone



class ann(BaseEstimator, ClassifierMixin):
    def __init__(self, no_trees=10, search_k=-1, num_neighbors=10,
                 metric='angular', weights='uniform'):
        self.model = None
        self.y = None

        self.search_k = search_k
        self.no_trees = no_trees
        self.num_neighbors = num_neighbors
        self.metric = metric
        self.weights = weights

    def fit(self, X, y, **fit_params):
        f = X.shape[1]
        self.model = AnnoyIndex(f, metric=self.metric)
        for ii, x_i in enumerate(X):
            self.model.add_item(ii, x_i)

        self.model.build(self.no_trees)

        self.y = y

        return self

    def predict(self, X, number=None, search_k=None, **pred_params):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X[None, :]

        if number is None:
            number = self.num_neighbors

        if search_k is None:
            search_k = self.search_k

        ret = np.array([self.predict1(X_i, number, search_k)
                        for X_i in X])
        return ret

    def predict1(self, X, number, search_k):
        if self.weights == 'distance':
            include_distances = True
            q, d = self.model.get_nns_by_vector(X, number, search_k, include_distances)
            d = np.array(d)
            w = (1/d) / np.sum(1/d)
        else:
            include_distances = False
            q = self.model.get_nns_by_vector(X, number, search_k, include_distances)
            w = np.ones(len(q)) / float(len(q))
        y_q = [self.y[q_i] for q_i in q]

        return np.dot(y_q, w)


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.svm import SVR, SVC
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.lda import LDA
    from sklearn.cross_validation import KFold

    from time import time

    from sklearn.datasets import load_boston
    boston = load_boston()
    X, y = boston.data, boston.target
    _, bins = np.histogram(y, bins=10)
    y_binned = np.digitize(y, bins=bins)

    clf = ann(num_neighbors=50, weights='distance')

    scores_baseline = []
    scores = []
    kf = KFold(len(y), n_folds=4)
    for train_ind, val_ind in kf:
        X_train, X_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]

        tic = time()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        toc = time() - tic

        score = mse(y_val, y_pred)
        print score
        scores.append(score)

        # q = ExtraTreesRegressor(n_estimators=400)
        q = KNeighborsRegressor(n_neighbors=50)
        q.fit(X_train, y_train)
        scores_baseline.append(mse(y_val, q.predict(X_val)))

    print 'Time:', toc
    print 'Scores \t\t\t', np.mean(scores)
    print 'KNN baseline \t', np.mean(scores_baseline)

