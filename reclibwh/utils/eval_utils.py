import pandas as pd
import numpy as np
import tensorflow as tf

def compute_auc(rec, test_user, train_user):

    rec_filt = np.setdiff1d(rec, train_user, assume_unique=True)

    n = len(rec_filt)
    p = len(test_user)

    if p == 0: # no positive examples to test against
        return -1

    f = n - p

    pos = np.isin(rec_filt, test_user)
    fp = np.nonzero(pos)[0].astype(float) - np.arange(len(test_user))  # np.cumsum((~pos).astype(np.float))
    fpr = np.cumsum(fp)[-1] / f
    return 1 - fpr/p

def filter_train_rec(rec, user_train):
    rec_filt = pd.DataFrame({'item': rec}, )
    rec_filt = pd.merge(rec_filt, user_train, on="item", how="outer", indicator=True)
    rec_filt = rec_filt.loc[rec_filt._merge == 'left_only']['item']
    return rec_filt

@tf.function
def tf_sort_scores(scores: tf.Tensor, axis: int):
    idx = tf.argsort(scores, axis=axis)
    return idx, tf.gather(scores, idx)