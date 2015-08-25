import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import itertools
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
import logging

from time import time
import sys


def shuffle_unison(a, b):
    """ Shuffles same-length arrays `a` and `b` in unison"""
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    return c[:, :a.size//len(a)].reshape(a.shape), c[:, a.size//len(a):].reshape(b.shape)


def get_transfer_inds(U_preds, tfer_threshs, tfer_weights):
    U_pred_avg = np.array(U_preds).mean(axis=0)  # Ensemble of base preds

    tfer_prob = ((U_pred_avg - tfer_threshs)/(1-tfer_threshs))*tfer_weights
    rng = np.random.rand(*tfer_prob.shape)
    winner = tfer_prob > rng

    inds = winner.sum(axis=1).astype(bool)
    U_y_pred = winner.argmax(axis=1)
    print U_pred_avg.max(axis=0)
    return inds, U_y_pred


class Rasco(BaseEstimator, ClassifierMixin):
    """ A Random Subspace Method for Co-Training
        A semi-supervised model for classification
    Ref:
        Wang, Jiao, Si-wei Luo, and Xian-hua Zeng.
        "A random subspace method for co-training."
        Neural Networks, 2008. IJCNN 2008.
        (IEEE World Congress on Computational Intelligence).
        IEEE International Joint Conference on. IEEE, 2008.
    """

    def __init__(self, h=None, feat_ratio=0.5, n_estimators=8, max_iters=20,
                 bootstrap=False,
                 verbose=False, log_handler=None):
        # todo: immutable arguments please
        """
        :param h: base estimator
        :param max_iters: max number of iterations
        :param n_estimators: number of sub classifiers to use
        :param feat_ratio: ratio of features to use per subspace
        :param bootstrap: Iterable of bootstrap resample `n` sample values to use
            per model. If `False`, don't bootstrap
        :param verbose: verbose logging
        :param log_handler: custom log handler can be passed in
        """
        self.log = logging.getLogger('rasco')
        if verbose:
            self.log.setLevel(logging.DEBUG)
            self.log.info('RASCO Init')
            if not self.log.handlers and log_handler:
                # pass
                # log_handler = logging.StreamHandler(sys.stdout)
                self.log.addHandler(log_handler)

        self.feat_ratio = feat_ratio
        self.n_estimators = n_estimators
        self.max_iters = max_iters

        self.bootstrap = bootstrap

        self.n_feats_subsp = None
        self.sub_sps_inds = None

        self.X_L = None
        self.X_U = None
        self.y_L = None
        self.classes_ = None
        self.h = h

        # self.estimators = [clone(h) for _ in range(self.n_estimators)]
        # self.estimators = None

    def get_transfers(self, preds):
        """
        :param preds: shape = (n_ests, n_obs, n_classes)
        """
        preds_avg = np.mean(preds, axis=0)
        y_preds = np.argmax(preds_avg, axis=1)
        ind = np.argmax(np.max(preds_avg, axis=1), 0)
        return [ind], y_preds[ind]

    def transfer(self, tfer_inds, y_tfer):
        X_tfer = self.X_U[tfer_inds, :]
        self.X_U = np.delete(self.X_U, list(tfer_inds), axis=0)  # Remove pt from set
        self.X_L, self.y_L = shuffle_unison(
            np.r_[self.X_L, X_tfer],
            np.r_[self.y_L, y_tfer],)

    def fit_init(self, X, y):
        self.estimators = [clone(h) for _ in range(self.n_estimators)]
        n_feats = X.shape[1]
        self.n_feats_subsp = self.feat_ratio * n_feats
        self.sub_sps_inds = [np.random.permutation(n_feats)[:self.n_feats_subsp]
                             for _ in range(self.n_estimators)]

        ind_u = np.isnan(y)
        self.X_L = X[~ind_u, :]
        self.X_U = X[ind_u, :]
        self.y_L = y[~ind_u]
        self.classes_ = np.unique(self.y_L)

    def fit_iter(self):
        preds = np.zeros((self.n_estimators, len(self.X_U), len(self.classes_)))
        for sub_i in range(self.n_estimators):
            X_L_sub = self.X_L[:, self.sub_sps_inds[sub_i]]
            self.estimators[sub_i].fit(X_L_sub, self.y_L)
            probs_pred = self.estimators[sub_i].predict_proba(self.X_U[:, self.sub_sps_inds[sub_i]])
            preds[sub_i, :, :] = probs_pred

        tfer_inds, y_tfer = self.get_transfers(preds)

        self.transfer(tfer_inds, y_tfer)

        return True

    def fit(self, X, y):
        """
        Unlabeled observations should have target value of NaN
        """
        # Setting up data
        self.fit_init(X, y)

        # Begin actual training
        start = time()
        for j in range(self.max_iters):
            self.log.debug('Iter %d' % j)
            tic = time()

            ret = self.fit_iter()

            if ret:
                pass
            else:
                break

            toc = time() - tic
            self.log.debug('Iter time: %g' % toc)

        end = time() - start
        self.log.debug('Total time: %g' % end)

    def predict_proba(self, X):
        return np.mean([est.predict_proba(X[:, sub_sp])
                        for sub_sp, est in zip(self.sub_sps_inds, self.estimators)],
                       axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

if __name__ == '__main__':
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_iris, make_circles
    from sklearn.cross_validation import KFold, StratifiedKFold

    iris = load_iris()
    X, y = iris.data, iris.target

    kf = KFold(n=len(y), n_folds=2)
    skf = StratifiedKFold(y, n_folds=2)

    scores = []
    scores_base = []
    for train_ind, test_ind in skf:
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        X_all = np.r_[X_train, X_test]
        y_all = np.r_[y_train, np.nan*np.ones(len(y_test))]

        h = LogisticRegression()
        clf = Rasco(h=h,
                    feat_ratio=0.5,
                    n_estimators=8,
                    max_iters=20,
                    verbose=True,
                    log_handler=logging.StreamHandler(sys.stdout))
        clf.fit(X_all, y_all)
        y_pred = clf.predict(X_test)

        scores.append(accuracy_score(y_test, y_pred))

        scores_base.append(accuracy_score(
            y_test, LogisticRegression().fit(X_train, y_train).predict(X_test)))

    print 'Rasco score:', scores
    print 'LogisticReg score:', scores_base
