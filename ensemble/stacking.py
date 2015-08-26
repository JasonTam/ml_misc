# from joblib import Parallel, delayed
# import multiprocessing
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_validation import KFold, StratifiedKFold

from collections import defaultdict

import time
import logging
import sys

def param_map(params):
    """
    :param params: keyed by a string in the following format:
        base#_method_paramname or meta_method_paramname
        # is the index of the base estimator
        method is either `fit` or `predict`
        or, for datasets, base#_X or base#_y
    :return:
    """
    split_d = defaultdict(lambda: {'fit': {}, 'predict': {}})
    for k, v in params.items():
        ind, param_name = k.split('_', 1)
        ind = ind.replace('base', '')
        if 'fit_' in param_name:
            split_d[ind]['fit'][param_name.split('fit_')[-1]] = v
        elif 'predict_' in param_name:
            split_d[ind]['predict'][param_name.split('predict_')[-1]] = v
        else:
            split_d[ind][param_name] = v
    return split_d


class Stacking(BaseEstimator, ClassifierMixin):
    """Stacked generalization
    Currently for regression
    """

    def __init__(self, base_estimators, meta_estimator,
                 cv=3, retrain=True, one_fold=False,
                 fit_params=None, pred_params=None,
                 include_orig_feats=False,
                 use_probs=True,
                 extra_data=None,
                 verbose=0,
                 log_handler=None,
                 save_level0_out=False):
        # todo: if a tuple if given for extradata, concatenate the datasets
        """
        :param base_estimators: base level 0 estimators
        :param meta_estimator: meta level 1 estimator
        :param cv: # of folds
        :param retrain: retrain base estimators on full set after meta is trained
        :param one_fold: only use 1 fold split
        :param fit_params: predefined fit params (so we don't have to pass them in when fitting)
        :param pred_params: predefined predict params (so we don't have to pass them in when fitting)
        :param include_orig_feats: include the original features for the meta estimator
        :param use_probs: use `predict_proba` on base models when available
            and use these probabilities as level0 output. This is only valid if the base model is a classifier
        :param extra_data: additional datasets or views that some base models will use
        :param verbose: verbose logging
        :param log_handler: custom log handler can be passed in
        :param save_level0_out: save the base preds and holdout targets
        :return:
        """
        self.log = logging.getLogger('stacker')
        if verbose:
            self.log.setLevel(logging.DEBUG)
            self.log.info('Stacker Init')
            if not self.log.handlers and log_handler:
                # pass
                # log_handler = logging.StreamHandler(sys.stdout)
                self.log.addHandler(log_handler)

        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.cv = cv

        self.retrain = retrain
        self.one_fold = one_fold

        self.fit_params = fit_params
        self.pred_params = pred_params
        self.include_orig_feats = include_orig_feats
        self.use_probs = use_probs

        if extra_data:
            self.extra_data = extra_data
            self.log.debug('Extra Data: %s' %
                           str(self.extra_data.keys()))
        else:
            self.extra_data = {}

        self.save_level0_out = save_level0_out
        self.level0_out = None

    def get_base_data(self, var, var_str, base_params):
        var_base_name = base_params.pop(var_str, None)
        var_base = self.extra_data[var_base_name] if var_base_name else var
        if var_base_name:
            self.log.debug('\t%s data: %s' % (var_str, var_base_name))
        return var_base

    def fit(self, X, y, **fit_params):
        # Parse params
        if not len(fit_params) and self.fit_params:
            fit_params = self.fit_params
        param_d = param_map(fit_params)

        holdout_base_preds = []
        holdout_xs = []
        holdout_ys = []
        kf = KFold(n=len(y), n_folds=self.cv)

        # KFold to make holdout sets for meta estimator
        for k_i, (train_ind, holdout_ind) in enumerate(kf):
            self.log.debug('===[Fit & predict fold %d for base models]===' % k_i)

            # train_ind, holdout_ind = iter(kf).next()
            X_train, X_holdout = X[train_ind], X[holdout_ind]
            y_train, y_holdout = y[train_ind], y[holdout_ind]
            holdout_base_preds_k = []
            # Train base estimators
            for ii, est in enumerate(self.base_estimators):
                est_name = str(est).split('(')[0]
                self.log.debug('[%s]' % est_name)

                # Setup special params & datasets
                base_params = param_d[str(ii)].copy()
                X_base = self.get_base_data(X, 'X', base_params)
                y_base = self.get_base_data(y, 'y', base_params)

                X_base_train, y_base_train = X_base[train_ind], y_base[train_ind]

                # Todo: need to append unlabeled points if there is a semi-supervised base method
                # Todo: ahhh this is bad
                nan_inds = np.isnan(y_base)
                X_base_train = np.r_[X_base_train, X_base[nan_inds]]
                y_base_train = np.r_[y_base_train, y_base[nan_inds]]

                X_base_holdout, y_base_holdout = X_base[holdout_ind], y_base[holdout_ind]

                # Fit base model
                tic = time.time()
                est.fit(X_base_train, y_base_train, **base_params['fit'])
                toc = time.time() - tic
                self.log.debug('\tTime fit:\t\t%g s' % toc)

                # Predict hold out set with base estimators
                # todo: include predict params from init
                tic = time.time()

                # Stupid stuff cuz keras wrapper doesnt comply with standards
                keras_reg_flag = False
                try:
                    keras_reg_flag = est.config_['layers'][-1]['output_dim'] == 1
                except (AttributeError, KeyError) as e:
                    pass

                if keras_reg_flag:
                    base_pred = est.predict_proba(X_base_holdout, **base_params['predict'])
                    toc = time.time() - tic
                    self.log.debug('\tTime predict_proba (keras):\t%g s' % toc)
                elif self.use_probs and hasattr(est, 'predict_proba'):
                    all_classes = list(np.unique(y_base))
                    base_pred_raw = est.predict_proba(X_base_holdout, **base_params['predict'])
                    toc = time.time() - tic
                    base_pred = np.zeros((len(base_pred_raw),
                                          len(all_classes)))
                    for pred, c in zip(base_pred_raw.T, est.classes_):
                        base_pred[:, all_classes.index(c)] = pred

                    self.log.debug('\tTime predict_proba:\t%g s' % toc)
                else:
                    base_pred = est.predict(X_base_holdout, **base_params['predict'])[:, None]
                    toc = time.time() - tic

                    self.log.debug('\tTime predict:\t\t%g s' % toc)

                holdout_base_preds_k.append(base_pred)

            holdout_base_preds.append(
                np.concatenate(holdout_base_preds_k, axis=1))

            holdout_xs.append(X_holdout)
            holdout_ys.append(y_holdout)
            if self.one_fold:
                break

        X_base_preds = np.concatenate(holdout_base_preds)
        y_base_preds = np.concatenate(holdout_ys)

        if self.save_level0_out:
            self.level0_out = (X_base_preds, y_base_preds)

        # --------------------[Meta Level]----------------------
        # Train meta estimator
        meta_params = param_d['meta']
        if self.include_orig_feats:
            X_meta = np.c_[X_base_preds, np.concatenate(holdout_xs)]
        else:
            X_meta = X_base_preds
        y_meta = y_base_preds
        self.meta_estimator.fit(X_meta, y_meta, **meta_params['fit'])

        # Retrain the base estimators on entire set
        if self.retrain:
            self.log.debug('Retraining base models on all data')
            for ii, est in enumerate(self.base_estimators):
                base_params = param_d[str(ii)].copy()
                X_base = self.get_base_data(X, 'X', base_params)
                y_base = self.get_base_data(y, 'y', base_params)

                est.fit(X_base, y_base, **base_params['fit'])

        return self

    def predict(self, X, **pred_params):
        # Parse params
        if not len(pred_params) and self.pred_params:
            pred_params = self.pred_params
        param_d = param_map(pred_params)

        base_preds = []
        for ii, est in enumerate(self.base_estimators):
            # Setup special params & datasets
            base_params = param_d[str(ii)].copy()
            X_base = self.get_base_data(X, 'X', base_params)

            # base_pred = est.predict(X, **base_params)
            if self.use_probs and hasattr(est, 'predict_proba'):
                base_pred = est.predict_proba(X_base, **base_params['predict'])
            else:
                base_pred = est.predict(X_base, **base_params['predict'])[:, None]

            base_preds.append(base_pred)

        base_preds = np.concatenate(base_preds, axis=1)

        meta_params = param_d['meta']
        if self.include_orig_feats:
            X_meta = np.c_[base_preds, X]
        else:
            X_meta = base_preds
        final_pred = self.meta_estimator.predict(X_meta, **meta_params['predict'])
        return final_pred


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.svm import SVR, SVC
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.lda import LDA

    from sklearn.datasets import load_boston
    boston = load_boston()
    X, y = boston.data, boston.target
    _, bins = np.histogram(y, bins=10)
    y_binned = np.digitize(y, bins=bins)

    base_ests = [
        LDA(),
        SVR(),
        KNeighborsRegressor(n_neighbors=20),
        # ExtraTreesRegressor(n_estimators=400),
                 ]
    meta_est = LinearRegression()

    scores_baseline = []
    scores = []
    kf = KFold(len(y), n_folds=4)
    for train_ind, val_ind in kf:
        X_train, X_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]
        y_binned_train, y_binned_val = y_binned[train_ind], y_binned[val_ind]
        stack = Stacking(base_ests, meta_est, cv=8,
                         include_orig_feats=False,
                         use_probs=True,
                         fit_params={
                             'base0_fit_tol': 1e-10,
                             'base0_y': 'y_binned',
                             'base2_X': 'X03_train',
                         },
                         pred_params={
                             'base2_X': 'X03_val',
                         },
                         extra_data={
                             'y_binned': y_binned_train,
                             'X03_train': X_train[:, 0:3],
                             'X03_val': X_val[:, 0:3],
                         },
                         verbose=True,
                         log_handler=logging.StreamHandler(sys.stdout),
                         save_level0_out=True,
                         )

        stack.fit(X_train, y_train)

        y_pred = stack.predict(X_val)
        score = mse(y_val, y_pred)
        print 'MSE:', score
        scores.append(score)

        # q = ExtraTreesRegressor(n_estimators=400)
        q = LinearRegression()
        q.fit(X_train, y_train)
        scores_baseline.append(mse(y_val, q.predict(X_val)))

    print
    print 'Score mean \t\t\t', np.mean(scores)
    print 'Score std \t\t\t', np.std(scores)
    print 'ET baseline \t', np.mean(scores_baseline)