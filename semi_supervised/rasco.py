import numpy as np



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