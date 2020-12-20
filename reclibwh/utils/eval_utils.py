import pandas as pd
import numpy as np
import tensorflow as tf

def compute_auc(rec, test_user, train_user):

    rec_filt = np.setdiff1d(rec, train_user, assume_unique=True)

    n = len(rec_filt)

    pos = np.isin(rec_filt, test_user)
    pos = np.nonzero(pos)[0]
    fp = pos.astype(float) - np.arange(len(pos))  # np.cumsum((~pos).astype(np.float))

    p = len(pos)
    if p == 0: return -1 # no positive examples to test against
    f = n - p
    if f==0: return 1 # every test item in rec

    fpr = np.cumsum(fp)[-1] / f
    return 1 - fpr/p

def compute_ap(rec, test_user, train_user):

    if len(test_user)==0: return -1

    rec_filt = np.setdiff1d(rec, train_user, assume_unique=True)

    n = len(rec_filt)

    pos = np.isin(rec_filt, test_user)
    pos = np.nonzero(pos)[0] + 1 # 1 indexed
    p = len(pos)

    if p == 0: return 0.
    pos = np.arange(1,len(pos)+1) / pos.astype(float)  # np.cumsum((~pos).astype(np.float))
    return np.sum(pos) / p

def filter_train_rec(rec, user_train):
    rec_filt = pd.DataFrame({'item': rec}, )
    rec_filt = pd.merge(rec_filt, user_train, on="item", how="outer", indicator=True)
    rec_filt = rec_filt.loc[rec_filt._merge == 'left_only']['item']
    return rec_filt

@tf.function
def tf_sort_scores(scores: tf.Tensor, axis: int):
    idx = tf.argsort(scores, axis=axis)
    return idx, tf.gather(scores, idx)