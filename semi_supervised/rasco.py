import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import itertools
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.base import clone

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
        A semi-supervised model for binary classification
    Ref:
        Wang, Jiao, Si-wei Luo, and Xian-hua Zeng.
        "A random subspace method for co-training."
        Neural Networks, 2008. IJCNN 2008.
        (IEEE World Congress on Computational Intelligence).
        IEEE International Joint Conference on. IEEE, 2008.
    """

    def __init__(self, h=None, feat_ratio, n_estimators=8, max_iters=20,
                 bootstrap=False,
                 verbose=False):
        # todo: immutable arguments please
        """
        :param h: base estimator
        :param max_iters: max number of iterations
        :param n_estimators: number of sub classifiers to use
        :param feat_ratio: ratio of features to use per subspace
        :param bootstrap: Iterable of bootstrap resample `n` sample values to use
            per model. If `False`, don't bootstrap
        :param verbose: verbosity (print timings etc)
        """
        self.verbose = verbose

        self.feat_ratio = feat_ratio
        self.n_estimators = n_estimators
        self.max_iters = max_iters

        self.bootstrap = bootstrap

        self.n_feats_subsp = None
        self.sub_sps_inds = None

        self.X_L = None
        self.X_U = None
        self.y_L = None

        self.estimators = [clone(h) for _ in range(self.n_estimators)]

    def fit_init(self, X, y):
        n_feats = np.unique(y)
        self.n_feats_subsp = self.feat_ratio * n_feats
        self.sub_sps_inds = [np.random.permutation(n_feats)[:self.n_feats_subsp]
                             for _ in range(self.n_estimators)]

        ind_u = np.isnan(y)
        self.X_L = X[~ind_u, :]
        self.X_U = X[ind_u, :]
        self.y_L = y[~ind_u]


    def fit_iter(self):
        preds = np.zeros()
        for sub_i in range(self.n_estimators):
            X_L_sub = self.X_L[:, self.sub_sps_inds[sub_i]]
            y_L_sub = self.y_L[self.sub_sps_inds[sub_i]]
            self.estimators[sub_i].fit(X_L_sub, y_L_sub)
            y_pred = self.estimators[sub_i].predict(self.X_U)
            preds.append(y_pred)


        pass


    def fit(self, X, y):
        """
        Unlabeled observations should have target value of NaN
        """
        # Setting up data
        self.fit_init(X, y)

        # Begin actual training
        start = time()
        for j in range(self.max_iters):
            print 'Iter %d' % j
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



sub_sps = [np.random.permutation(n_feats)[:m]
               for k in range(K)]

score_list = []
score_fn = roc_auc_score
transfer_counts = []
for j in range(J):
    print 'Iter # %d' % j
#     print 'Winner thresh:', prob_thresh
    print 'Tfer thresh:', tfer_threshs
    print 'Tfer weights:', tfer_weights
    # Subspace creation
    val_preds = []

    ret_iter = Parallel(n_jobs=-1)(
        delayed(train_predict_model)(
            L_X[:, sub_sp], L_y,
            U_X[:, sub_sp],
            X_valid[:, sub_sp],
            s_i,
        )
        for s_i, sub_sp in enumerate(sub_sps))

    models, U_preds, val_preds = zip(*ret_iter)

    # Getting the current score for validation set
    val_pred_ens = np.array(val_preds).mean(axis=0)
    score_ens = score_fn(y_valid, val_pred_ens[:, 1])
    score_list.append(score_ens)
    print 'Ensemble score:', score_ens

    # Current predictions for test set
#     U_pred_avg = np.array(U_preds).mean(axis=0)  # Ensemble of base preds
#     max_probs = U_pred_avg.max(axis=1)

#     U_y_pred = np.argmax(U_pred_avg, axis=1)

#     win_inds = max_probs > prob_thresh
    win_inds, U_y_pred = get_transfer_inds(U_preds, tfer_threshs, tfer_weights)

    L_X, L_y = shuffle_unison(
            np.r_[L_X, U_X[win_inds, :]],
            np.r_[L_y, U_y_pred[win_inds]],)

    U_X = np.delete(U_X, np.where(win_inds), axis=0)


    print 'Transferred unlabeled observations: %d' % win_inds.sum()
    tfer_dist = [list(U_y_pred[win_inds]).count(cc)
                 for cc in range(n_classes)]
    print 'Transfer distribution:', [list(U_y_pred[win_inds]).count(cc)
                                     for cc in range(n_classes)]
    transfer_counts.append(tfer_dist)
    print 'Cumulative transfer:', np.array(transfer_counts).sum(axis=0)

    for ii, nt in enumerate(tfer_dist):
        if nt:
            tfer_weights[ii] = tfer_weights[ii]*(1+1./nt)/2.
#     if not win_inds.sum():
#         prob_thresh -= thresh_decay

    sys.stdout.flush()