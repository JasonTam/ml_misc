import numpy as np
from time import time
from sklearn.neighbors import KNeighborsRegressor
import sys

from sklearn.base import BaseEstimator, ClassifierMixin, clone

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVR

from rpforest import RPForest


def top_k(arr, k):
    return np.argsort(arr)[-k:][::-1]

def get_neighbors(x, nn_tracker):
    if hasattr(nn_tracker, 'kneighbors'):
        return nn_tracker.kneighbors(x, return_distance=False)
    elif hasattr(nn_tracker, 'query'):
        return nn_tracker.query(x)
    else:
        raise AttributeError

def calc_d(clf_type, x_u, h, L, y, nn_tracker, **clf_params):
    """
    Calculates the delta, the difference in squared errors between
        the regressor with and without the additional pseudolabeled points
    :param clf_type: type of model for new regressor
    :param x_u: features of unlabeled observations
    :param h: the current regressor
    :param L: features of labeled observations
    :param y: true targets of the observations in `L`
    :param clf_params: params for new model
    :return: delta MSE
    """
    # Get pseudo labels
    y_est = h.predict(x_u)

    # Set of (indices) k nearest neighbors of x_u
    # measuring MSE on the entire labeled set is time consuming
    # COREG makes the approximation of d_xu by using only
    # the neighbors to calculate MSE
    Omega = get_neighbors(x_u, nn_tracker)

    # New regressor w/ additional info
    # _p denotes 'prime' tick in paper (for the new regressor)
    h_p = clf_type(**clf_params)
    h_p.fit(np.r_[L, x_u[None, :]], np.r_[y, y_est])

    # Compare MSE (use Omega to select only the neighbors)
    y_i = y[Omega[0]]
    x_i = L[Omega[0], :]

    # Get squared errors (se)
    se_o = (y_i - h.predict(x_i))**2        # mse for orig
    se_p = (y_i - h_p.predict(x_i))**2      # mse for new

    # Difference of squared errors
    d_xu = (se_o - se_p).sum(axis=0)

    return d_xu

class CoReg(BaseEstimator, ClassifierMixin):
    """ A wrapper for CoReg: co-training for regression
    References:
        Zhou, Zhi-Hua, and Ming Li.
        "Semi-Supervised Regression with Co-Training."
        IJCAI. Vol. 5. 2005.
    """

    def __init__(self, h=None, n_u=25000, T=20, k=1,
                 verbose=False, n_jobs=-1):
        # todo: immutable arguments please
        """
        :param h: iterable of regressor models to use in the coreg process
        :param n_u: number of unlabeled observations to use (screw it)
        :param T: number of iterations to perform (# of observations to transfer)
        :param k: number of top candidates to transfer
        :param verbose: verbosity (print timings etc)
        :param n_jobs: # jobs to run in parallel.
            Note that many cases parallelizing will introduce too much
            overhead to be faster than using just a single thread.
        """
        self.verbose = verbose
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

        self.n_u = n_u
        self.T = T
        self.k = k

        self.X_L = None  # Original labeled set
        self.L = None    # Iterable of the current trainings sets for each model
        self.y_train = None  # Original targets
        self.y = None   # Iterable of current targets (should correspond to `L`)
        self.X_U = None  # Original unlabeled set
        self.U = None    # Current unlabeled set
        self.U_p = None  # Shuffled and sampled unlabeled set

        if h is None:  # Default to KNeighborsRegressor if nothing is given
            self.h = [KNeighborsRegressor(n_neighbors=k_j, p=p_j)
                      for L_j, y_j, k_j, p_j in zip(self.L, self.y,
                                                    [3, 3], [2, 5])]
        else:
            self.h = h

        # Used for tracking NN
        self.nn_trackers = [RPForest(leaf_size=50, no_trees=10),
                            RPForest(leaf_size=50, no_trees=10)]

    def fit_init(self, X, y):
        # Setting up data
        ind_u = np.isnan(y)
        self.X_L = X[~ind_u, :]
        self.y_train = y[~ind_u]
        self.X_U = X[ind_u, :]

        self.L = [self.X_L, self.X_L]
        self.y = [self.y_train, self.y_train]

        self.U_p = self.X_U[np.random.permutation(
            np.arange(len(self.X_U)))[:self.n_u], :]

        # Fitting base models
        for ii in range(len(self.h)):
            self.h[ii].fit(self.L[ii], self.y[ii])
            self.nn_trackers[ii].fit(self.L[ii])

    def fit(self, X, y):
        """
        Unlabeled observations should have target value of NaN
        """
        # Setting up data
        self.fit_init(X, y)

        # Begin actual training
        start = time()
        for t in range(self.T):
            print t,
            tic = time()
            # `pi` will hold the unlabeled observations to add to training set
            pi = [None] * len(self.h)
            for j in range(len(self.h)):

                # List of MSE diffs (per unlabeled obs)
                clf_params = self.h[j].get_params()
                clf_type = type(self.h[j])
                if self.n_jobs > 1:
                    d_xu_l = Parallel(n_jobs=self.n_jobs, backend='threading')(
                        delayed(calc_d)(clf_type, x_u,
                                        self.h[j], self.L[j], self.y[j],
                                        self.nn_trackers[j],
                                        **clf_params)
                        for x_u in self.U_p)
                else:
                    d_xu_l = [calc_d(clf_type, x_u,
                                     self.h[j], self.L[j], self.y[j],
                                     self.nn_trackers[j],
                                     **clf_params)
                              for x_u in self.U_p]

                d_xu_l = np.array(d_xu_l)
                if any(d_xu_l > 0):  # At least one obs made an improvement
                    ind_top = top_k(d_xu_l, self.k)  # Index of top k candidates
                    x_top = self.U_p[ind_top, :]
                    y_top = self.h[j].predict(x_top)
                    pi[j] = (x_top, y_top)    # New pt to be added next iter
                    self.U_p = np.delete(self.U_p, (ind_top), axis=0)    # Remove pt from set
                else:
                    pi[j] = None

            change_flag = False     # If something was added to the train set
            for j in range(len(self.h)):
                if pi[j]:
                    ii = (j + 1) % 2  # To get the other regressor
                    # Add new pt to the OTHER regressor's training
                    self.L[ii] = np.append(self.L[ii], pi[j][0], axis=0)
                    self.y[ii] = np.append(self.y[ii], pi[j][1])
                    self.h[ii].fit(self.L[ii], self.y[ii])

                    self.nn_trackers[ii].trees = []
                    self.nn_trackers[ii].fit(self.L[ii])
                    change_flag = True

            if change_flag:
                # Replenish U_p
                # screw it
                pass
            else:
                break
            toc = np.round(time() - tic, 2)
            if self.verbose:
                print 'Iteration time:', toc,
                sys.stdout.flush()

        end = time() - start
        if self.verbose:
            print 'Total time:', end

    def predict(self, X):
        return np.mean([h.predict(X) for h in self.h], axis=0)



if __name__ == '__main__':
    import xgboost as xgb
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

        # h = [KNeighborsRegressor(n_neighbors=3, p=2),
        #      xgb.XGBRegressor()]
        h = [xgb.XGBRegressor(),
             ElasticNet()]
        clf = CoReg(h=h, T=10, verbose=True, k=5, n_jobs=1)
        clf.fit(X_all, y_all)
        y_pred = clf.predict(X_test)

        scores.append(mean_squared_error(y_test, y_pred))

        # clf_base = KNeighborsRegressor(n_neighbors=3, p=2)
        clf_base = xgb.XGBRegressor()
        clf_base.fit(X_train, y_train)
        y_pred_base = clf_base.predict(X_test)
        scores_base.append(mean_squared_error(y_test, y_pred_base))

    print 'Coreg score:', scores
    print 'Base score:', scores_base




