from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def fit_bin(X, y,
            est=None, est_type=None, est_params=None,
            label_transformer=None,
            **fit_params):
    """ Fits a new estimator given an estimator object or estimator type and params
    :param est: estimator object
    :param est_type: estimator type
    :param est_params: estimator params
    :param X: feature vectors
    :param y: targets
    :param fit_params:
    :return: new fitted estimator
    """
    # Todo: for some reason the ensemble models have boolean value as False
    # assert any([est, est_type])

    if label_transformer:
        y = label_transformer(y)

    new_est = None
    if est is not None:
        new_est = clone(est)
    elif est_type:
        new_est = est_type(**est_params)

    return new_est.fit(X, y, **fit_params)


class SimpleOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """A wrapper for ordinal classification
    References:
        Frank, Eibe, and Mark Hall. 
        A simple approach to ordinal classification. 
        Springer Berlin Heidelberg, 2001.
    
    The discrete encoded classes should have ordinal values
    """
    
    def __init__(self, base_estimator=None,
                 base_estimator_type=None, base_estimator_params=None,
                 label_transformer=None,
                 nan_support=True,
                 n_jobs=-1):
        """
        :param base_estimator: base binary estimator object
        :param base_estimator_type: estimator type and params can be given instead of `base_estimator`
        :param base_estimator_params: params to initialize `base_estimator_type` if given
        :param label_transformer: transformation to apply to labels before binary split
            (ex. transforming to categorical format)
        :param nan_support: support for NaN targets for semi-supervised base estimators
        :param n_jobs: # of threads to run
        :return:
        """
        self.base_estimator = base_estimator
        self.base_estimator_type = base_estimator_type if base_estimator_type else {}
        self.base_estimator_params = base_estimator_params
        self.label_transformer = label_transformer
        self.nan_support = nan_support

        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

        self.classes_ = None
        self.k_ = None
        self.estimators_ = None

    def fit(self, X, y, **fit_params):
        self.classes_ = np.unique(y[~np.isnan(y)])
        self.k_ = len(self.classes_)
        
        # Fitting binary classifiers at each cut
        # new binary targets
        if self.nan_support:
            c_bins = []
            for cut in sorted(self.classes_)[:-1]:
                y_bin = (y > cut).astype(float)
                y_bin[np.isnan(y)] = np.nan
                c_bins.append(y_bin)
        else:
            c_bins = [y > cut for cut in sorted(self.classes_)[:-1]]

        if self.n_jobs > 1:
            self.estimators_ = Parallel(n_jobs=self.n_jobs, backend='threading')\
                (delayed(fit_bin)(X, c_bin,
                                  self.base_estimator, self.base_estimator_type, self.base_estimator_params,
                                  self.label_transformer,
                                  **fit_params)
                 for c_bin in c_bins)
        elif self.n_jobs == 1:
            self.estimators_ = []
            for c_bin in c_bins:
                fitted_clf = fit_bin(X, c_bin,
                                     self.base_estimator, self.base_estimator_type, self.base_estimator_params,
                                     self.label_transformer,
                                     **fit_params)
                self.estimators_.append(fitted_clf)

        return self
    
    def predict_proba(self, X, **pred_params):
        """Probs corresponding to self.classes_"""
        p = np.array([clf.predict_proba(X, **pred_params)[:, 1]
              for clf in self.estimators_]).T
        pr = -np.diff(np.c_[np.ones(p.shape[0]), p, np.zeros(p.shape[0])],
                      axis=1)
        return pr
    
    def predict(self, X, weighted=False, **pred_params):
        if weighted:
            return self.predict_weighted(X, **pred_params)
        else:
            max_inds = np.argmax(self.predict_proba(X, **pred_params), axis=1)
            return np.array([self.classes_[ind] for ind in max_inds])

    def predict_weighted(self, X, geometric=False, **pred_params):
        probs = self.predict_proba(X, **pred_params)
        if geometric:
            return np.exp(
                np.sum(probs*np.log(self.classes_), axis=1) /
                np.sum(probs, axis=1))
        else:
            return np.sum(self.classes_ * probs, axis=1)

