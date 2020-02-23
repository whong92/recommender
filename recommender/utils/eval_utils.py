import pandas as pd
import numpy as np
from keras.callbacks import Callback
from .ItemMetadata import ExplicitDataFromCSV
from tqdm import tqdm
from recommender.recommender.recommenderInterface import Recommender
from keras.utils import generic_utils
import tensorflow as tf
from typing import Callable, Any, Union, Iterable

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

def compute_ap(model, user, test, train):

    pos = set(test[test['user'] == user]['item'])
    rec = model.recommend(user)[0]
    df_test_excl = train.loc[train.user == user]
    rec_filt = filter_train_rec(rec, df_test_excl)

    p = len(pos)

    if p == 0: # no positive examples to test against
        return -1

    tp = 0
    ap = 0

    for i, r in enumerate(rec_filt):
        if r in pos:
            tp += 1
            ap += tp/(i+1)
    return ap/p

def filter_train_rec(rec, user_train):
    rec_filt = pd.DataFrame({'item': rec}, )
    rec_filt = pd.merge(rec_filt, user_train, on="item", how="outer", indicator=True)
    rec_filt = rec_filt.loc[rec_filt._merge == 'left_only']['item']
    return rec_filt

@tf.function
def tf_sort_scores(scores: tf.Tensor, axis: int):
    idx = tf.argsort(scores, axis=axis)
    return idx, tf.gather(scores, idx)

def eval_model(model: Recommender, data: ExplicitDataFromCSV, batchsize:int=10, M:Union[int, Iterable[int]]=100):

    if type(M) is int:
        M = np.arange(0,M,dtype=int)

    AUC = -np.ones(shape=(M.shape[0],))
    progbar = generic_utils.Progbar(M.shape[0])

    for m in range(0,M.shape[0],batchsize):
        t = min(M.shape[0], m + batchsize)
        recs = model.recommend(M[m:t])[0]
        for i, rec in enumerate(recs):
            df_train, df_test = data.get_user_ratings(m+i)
            df_test_rel = np.array(
                df_test.loc[df_test['rating'] > 3.0, 'item'])
            df_train = np.array(df_train['item'])
            auc = compute_auc(rec, df_test_rel, df_train)

            AUC[m+i] = auc
        progbar.add(len(recs), values=[('AUC', np.mean(AUC[m:t]))])

    return AUC

class AUCCallback(Callback):

    def __init__(self, data: ExplicitDataFromCSV, M:Union[int, Iterable[int]]=100, batchsize:int=10, save_fn:Callable=None):
        super(AUCCallback, self).__init__()
        self.data = data
        self.M = M
        self.batchsize=batchsize
        self.save_fn = save_fn
        self.best_AUC = 0.
        self.AUCe = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        AUC = eval_model(self.model, self.data, M=self.M, batchsize=self.batchsize)
        self.AUCe.append(np.mean(AUC[AUC>-1]))
        self.epochs.append(epoch)
        if self.AUCe[-1] >= self.best_AUC and self.save_fn is not None:
            self.save_fn()
            self.best_AUC = self.AUCe[-1]

    def save_result(self, outfile):
        pd.DataFrame({'epoch': self.epochs, 'AUC': self.AUCe}).to_csv(outfile, index=False)