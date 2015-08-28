import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import itertools
from joblib import Parallel, delayed
import multiprocessing
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


def fit_sub(sub_inds, X_L, y_L, X_U, est):
    """ Fits a particular subspace and predicts unlabeled observations
    :param sub_inds: feature inds to create subspace
    :param X_L: feature corresponding to labeled observations
    :param y_L: targets of labeled observations
    :param X_U: features of unlabeled observations
    :param est: estimator to use
    :return: prediction probabilities of unlabeled observations
    """
    X_L_sub = X_L[:, sub_inds]
    est.fit(X_L_sub, y_L)
    probs_pred = est.predict_proba(X_U[:, sub_inds])
    return probs_pred


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

    def __init__(self, base_estimator=None, feat_ratio=0.5, n_estimators=8, max_iters=20,
                 n_xfer=1,
                 y_validation=None,
                 n_jobs=1,
                 verbose=False, log_handler=None):
        # todo: immutable arguments please
        """
        :param base_estimator: base estimator
        :param max_iters: max number of iterations
        :param n_estimators: number of sub classifiers to use
        :param feat_ratio: ratio of features to use per subspace
        :param y_validation: labels for unlabeled obs for the sake of validation
        :param n_jobs: number of jobs to run in parallel
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
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

        self.feat_ratio = feat_ratio
        self.n_estimators = n_estimators
        self.max_iters = max_iters
        self.n_xfer = n_xfer

        self.n_feats_subsp = None
        self.sub_sps_inds = None

        self.X_L = None
        self.X_U = None
        self.y_L = None
        self.classes_ = None
        self.base_estimator = base_estimator

        # self.estimators = [clone(h) for _ in range(self.n_estimators)]
        self.estimators = None

        self.y_val = y_validation

    def get_transfers(self, preds):
        """
        :param preds: shape = (n_ests, n_obs, n_classes)
        """
        preds_avg = np.mean(preds, axis=0)
        y_preds = np.argmax(preds_avg, axis=1)
        maxes = np.max(preds_avg, axis=1)
        # ind = np.argmax(maxes, 0)   # n_xfer=1
        ind = maxes.argsort()[-self.n_xfer:][::-1]

        if self.y_val is None:
            self.log.debug('Best candidate: pred_class=%s | Prob: %s'
                           % (str(y_preds[ind]), maxes[ind])
                           )
        else:
            self.log.debug('Best candidate: pred=%g true=%g | Prob: %g'
                           % (y_preds[ind], self.y_val[ind], np.max(np.max(preds_avg, axis=1), 0)))
        return ind, y_preds[ind]

    def transfer(self, tfer_inds, y_tfer):
        X_tfer = self.X_U[tfer_inds, :]
        self.X_U = np.delete(self.X_U, list(tfer_inds), axis=0)  # Remove pt from set
        if self.y_val is not None:
            self.y_val = np.delete(self.y_val, list(tfer_inds))
        self.X_L, self.y_L = shuffle_unison(
            np.r_[self.X_L, X_tfer],
            np.r_[self.y_L, y_tfer],)
        self.log.debug('L-shape: %s \t U-shape: %s'
                       % (str(self.X_L.shape), str(self.X_U.shape)))

    def fit_init(self, X, y):
        if isinstance(self.base_estimator, list):
            self.estimators = self.base_estimator
            self.n_estimators = len(self.estimators)
        else:
            self.estimators = [clone(self.base_estimator) for _ in range(self.n_estimators)]
        n_feats = X.shape[1]
        self.n_feats_subsp = np.round(self.feat_ratio * n_feats)
        self.sub_sps_inds = [np.random.permutation(n_feats)[:self.n_feats_subsp]
                             for _ in range(self.n_estimators)]

        ind_u = np.isnan(y)
        self.X_L = X[~ind_u, :]
        self.X_U = X[ind_u, :]
        self.y_L = y[~ind_u]
        self.classes_ = np.unique(self.y_L)

    def fit_iter(self):
        preds_list = Parallel(n_jobs=self.n_jobs, backend='threading')\
            (delayed(fit_sub)(sub_inds, self.X_L, self.y_L, self.X_U, est)
             for sub_inds, est in zip(self.sub_sps_inds, self.estimators))
        preds = np.array(preds_list)

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
            tic = time()

            ret = self.fit_iter()

            if ret:
                pass
            else:
                break

            toc = time() - tic
            self.log.debug('Iter #%d time: %g' % (j, toc))

        end = time() - start
        self.log.debug('Total time: %g' % end)

        return self

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
    from sklearn.svm import SVR, SVC
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

        # h = LogisticRegression()
        h = [LogisticRegression() for _ in range(8)]
        clf = Rasco(base_estimator=h,
                    feat_ratio=0.5,
                    n_estimators=8,
                    max_iters=20,
                    y_validation=y_test,
                    n_jobs=1,
                    verbose=True,
                    log_handler=logging.StreamHandler(sys.stdout))
        clf.fit(X_all, y_all)
        y_pred = clf.predict(X_test)

        scores.append(accuracy_score(y_test, y_pred))

        scores_base.append(accuracy_score(
            y_test, LogisticRegression().fit(X_train, y_train).predict(X_test)))

    print 'Rasco score:', scores
    print 'LogisticReg score:', scores_base
