import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import itertools
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR

from time import time
import sys


def shuffle_unison(a, b):
    """ Shuffles same-length arrays `a` and `b` in unison"""
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    return c[:, :a.size//len(a)].reshape(a.shape), c[:, a.size//len(a):].reshape(b.shape)


def bootstrap_resample(X=None, n=None, inds=False):
    """ Bootstrap resampling
    :param X: observations to resample
    :param n: Number of samples to pick out.
        If `None`, same number of samples as original observations.
        If `n` < 1, `n` will be treated as a proportion
    :param inds: If `True` just return the resampled indices
    :return:
    """
    if n is None:
        n = len(X)
    elif 0 < n < 1:
        n = int(len(X) * n)
    else:
        raise ValueError('Invalid `n`')

    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    if inds:
        return resample_i
    else:
        X_resample = X[resample_i]
        return X_resample


class TriReg(BaseEstimator, ClassifierMixin):
    """ Based on Tri-training for classification
    Ref:
        Zhou, Zhi-Hua, and Ming Li.
        "Tri-training: Exploiting unlabeled data using three classifiers."
        Knowledge and Data Engineering, IEEE Transactions on 17.11 (2005): 1529-1541.

        Zhou, Zhi-Hua, and Ming Li.
        "Semi-supervised learning by disagreement."
        Knowledge and Information Systems 24.3 (2010): 415-439.
    """

    def __init__(self, h=None, T=10, accept_thresh=0.5,
                 bootstrap=False,
                 verbose=False):
        # todo: immutable arguments please
        """
        :param h: iterable of regressor models to use tri-training process
        :param T: number of iterations
        :param bootstrap: Iterable of bootstrap resample `n` sample values to use
            per model. If `False`, don't bootstrap
        :param verbose: verbosity (print timings etc)
        """
        self.verbose = verbose
        self.T = T
        self.accept_thresh = accept_thresh
        self.bootstrap = bootstrap

        self.L_X = None    # Iterable of the current trainings sets for each model
        self.L_y = None   # Iterable of current targets (should correspond to `L_X`)
        self.U_X = None  # Unlabeled set

        self.tot_xfer = 0  # Cumulative transfer
        self.n_u_orig = None    # Original number of unlabeled observations

        if h is None:  # Default to KNeighborsRegressor if nothing is given
            self.h = [KNeighborsRegressor(n_neighbors=k_j, p=p_j)
                      for k_j, p_j in zip([3, 3, 8], [2, 5, 1])]
        else:
            self.h = h
        assert(len(self.h) == 3)

    def fit_init(self, X, y):
        # Setting up data
        ind_u = np.isnan(y)
        X_L = X[~ind_u, :]
        y_train = y[~ind_u]
        # self.X_U = X[ind_u, :]

        # Each model has its own dedicated pool of labeled data
        if self.bootstrap:
            boot_inds = [bootstrap_resample(X=X_L, n=self.bootstrap[ii], inds=True)
                         for ii in range(len(self.h))]
            self.L_X = [X_L.copy()[inds] for inds in boot_inds]
            self.L_y = [y_train.copy()[inds] for inds in boot_inds]
        else:
            self.L_X = [X_L.copy()] * len(self.h)
            self.L_y = [y_train.copy()] * len(self.h)

        # Every model shares the same pool of unlabeled data to grab from
        self.U_X = X[ind_u, :]
        self.n_u_orig = len(self.U_X)

    def fit_iter(self):
        win_list = []
        updata = np.zeros(len(self.h))
        U_preds = np.zeros((len(self.U_X), len(self.h)))

        for ii in range(len(self.h)):
            self.h[ii].fit(self.L_X[ii], self.L_y[ii])      # todo: add **fit_params
            # print 'Model: %d | %s' % (ii, str(self.h[ii])[:13])

            # TODO: Track validation loss etc here

            # Predict unlabeled data
            if len(self.U_X):
                U_preds[:, ii] = self.h[ii].predict(self.U_X)
            else:
                print 'Ran out of unlabeled obs'
                return 0
        transfers = get_transfers(U_preds, thresh=self.accept_thresh)
        num_xfer = self.transfer_obs(transfers)
        self.tot_xfer += num_xfer
        print 'Number transfers: %d (%d%%)' % (num_xfer, 100.*num_xfer/len(U_preds)),
        print 'Total transfers: %d/%d (%d%%)' % (self.tot_xfer, self.n_u_orig, 100.*self.tot_xfer/self.n_u_orig)
        return num_xfer

    def transfer_obs(self, transfers):
        inds_remove_l = []
        for ii, transfer in enumerate(transfers):
            inds, vals = transfer

            # Add obs to labeled set
            self.L_X[ii], self.L_y[ii] = shuffle_unison(
                np.r_[self.L_X[ii], self.U_X[inds, :]],
                np.r_[self.L_y[ii], vals],)

            # Collect inds to remove
            inds_remove_l.append(inds)

        # Remove from unlabeled set
        inds_remove = set(itertools.chain(*inds_remove_l))
        self.U_X = np.delete(self.U_X, list(inds_remove), axis=0)

        return len(inds_remove)

    def fit(self, X, y):
        """
        Unlabeled observations should have target value of NaN
        """
        # Setting up data
        self.fit_init(X, y)

        # Begin actual training
        start = time()
        for t in range(self.T):
            print 'Iter %d' % t
            tic = time()

            ret = self.fit_iter()

            if ret:
                # Replenish U_p
                # screw it
                pass
            else:
                break

            toc = np.round(time() - tic, 2)
            if self.verbose:
                print 'Iteration time:', toc
                sys.stdout.flush()

        end = time() - start
        if self.verbose:
            print 'Total time:', end

    def predict(self, X):
        return np.mean([h.predict(X) for h in self.h], axis=0)


def get_transfers(U_preds, thresh=0.5):
    """ Criterion for transferring unlabeled data
    :param U_preds:
    :return list of transfers. Each transfer is a tuple (index, value)
    For tri-reg, the length of the list is 3:
        (to L_1, to L_2, to L_3)
    """
    transfers = []
    dist = lambda a, b: np.abs(a - b)
    acceptance = lambda a, b: dist(a, b) < thresh

    cols = set(range(U_preds.shape[1]))
    for col in cols:
        cols_other = list(cols-{col})
        col_vals = (U_preds[:, cols_other[0]], U_preds[:, cols_other[1]])

        inds_xfer = np.where(acceptance(*col_vals))[0]

        # Pseudo label will be the mean of the agreeing models' predictions
        vals_xfer = np.mean(col_vals, axis=0)[inds_xfer]

        # todo: introduce some stoachstic acceptance layer

        transfers.append((inds_xfer, vals_xfer))
    return transfers



if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston
    boston = load_boston()
    X, y = boston.data, boston.target

    from sklearn.cross_validation import KFold

    kf = KFold(n=len(y), n_folds=2)

    scores = []
    scores_base = []
    for train_ind, test_ind in kf:
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        X_all = np.r_[X_train, X_test]
        y_all = np.r_[y_train, np.nan*np.ones(len(y_test))]


        h = [LinearRegression(), ElasticNet(), KNeighborsRegressor()]

        clf = TriReg(h=h, T=10, verbose=True)
        clf.fit(X_all, y_all)
        y_pred = clf.predict(X_test)

        scores.append(mean_squared_error(y_test, y_pred))
        # scores_base.append(mean_squared_error(
        #     y_test, LogisticRegression().fit(X_train, y_train).predict(X_test)))
        scores_base.append(mean_squared_error(
            y_test, SVR().fit(X_train, y_train).predict(X_test)))

    print 'Trireg score:', scores
    print 'Logreg score:', scores_base


