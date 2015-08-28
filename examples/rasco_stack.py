import sys
import logging
import numpy as np
from sklearn.cross_validation import KFold
from ensemble.stacking import Stacking
from semi_supervised.rasco import Rasco


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.svm import SVR, SVC
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.lda import LDA

    from sklearn.datasets import load_boston

    boston = load_boston()
    X, y = boston.data, boston.target
    _, bins = np.histogram(y, bins=10)
    y_binned = np.digitize(y, bins=bins) - 1

    rasco = Rasco(base_estimator=[DecisionTreeClassifier() for _ in range(8)],
                  feat_ratio=0.5,
                  n_estimators=8,
                  max_iters=5,
                  n_xfer=2,
                  n_jobs=1,
                  verbose=True,
                  log_handler=logging.StreamHandler(sys.stdout))

    base_ests = [
        rasco,
        SVR(),
        KNeighborsRegressor(n_neighbors=20),
        # ExtraTreesRegressor(n_estimators=400),
    ]
    meta_est = SVR()

    scores_baseline = []
    scores = []
    kf = KFold(len(y), n_folds=2)
    for train_ind, val_ind in kf:
        X_train_k, X_val_k = X[train_ind], X[val_ind]
        y_train_k, y_val_k = y[train_ind], y[val_ind]
        y_binned_train_k, y_binned_val_k = y_binned[train_ind], y_binned[val_ind]
        X_all_k = np.r_[X_train_k, X_val_k]
        y_all_k = np.r_[y_binned_train_k, np.nan * np.ones(len(y_binned_val_k))]

        # rasco.y_val = y_binned_val_k

        stack = Stacking(base_ests, meta_est, cv=8,
                         include_orig_feats=False,
                         use_probs=True,
                         fit_params={
                             'base0_X': 'X_all',
                             'base0_y': 'y_all',
                         },
                         pred_params={
                         },
                         extra_data={
                             'X_all': X_all_k,
                             'y_all': y_all_k,
                             'y_binned': y_binned_train_k,
                         },
                         verbose=True,
                         log_handler=logging.StreamHandler(sys.stdout),
                         save_level0_out=True,
                         )

        stack.fit(X_train_k, y_train_k)

        y_pred = stack.predict(X_val_k)
        score = mse(y_val_k, y_pred)
        print 'MSE:', score
        scores.append(score)

        # q = ExtraTreesRegressor(n_estimators=400)
        q = LinearRegression()
        q.fit(X_train_k, y_train_k)
        scores_baseline.append(mse(y_val_k, q.predict(X_val_k)))

    print
    print 'Score mean \t\t\t', np.mean(scores)
    print 'Score std \t\t\t', np.std(scores)
    print 'ET baseline \t', np.mean(scores_baseline)