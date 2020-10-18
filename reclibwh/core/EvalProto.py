from abc import ABC, abstractmethod
from .Environment import EvalProto, Algorithm, Environment
from .RecAlgos import RecAlgo
from ..utils.eval_utils import compute_auc
from keras.utils import generic_utils
import numpy as np
import os
# from keras.callbacks import Callback
from tensorflow.keras.callbacks import Callback
import pandas as pd
from .Environment import RecAlgo

# generic evaluation protocols

class AUCEval(EvalProto):

    """
    requires that the encapsulating environment object has a 'rec_algo' recommendation
    algorithm available
    """

    def __init__(self, env, rec: RecAlgo, med_score):
        self.__env = env
        self.__med_score = med_score
        self.__rec = rec

    def evaluate(self):

        rec = self.__rec
        data = self.__env.get_state()['data']['auc_data']
        train_data, test_data = data['train'], data['test']
        N = len(train_data)*train_data.row_batch_size
        AUC = -np.ones(shape=(N,))
        progbar = generic_utils.Progbar(N)

        m = 0
        for train_batch, test_batch in zip(train_data, test_data):

            i_train, pad_val = train_batch['cols'], train_batch['pad_val']
            users, i_test, r_test = test_batch['rows'], test_batch['cols'], test_batch['val']

            recs, _ = rec.recommend(users)

            for i, items in enumerate(recs):

                i_test_rel = i_test[i][i_test[i] != pad_val]
                i_test_rel = i_test_rel[i_test_rel >= self.__med_score]
                i_train_rel = i_train[i][i_train[i] != pad_val]

                auc = compute_auc(items, i_test_rel, i_train_rel)
                if auc < 0: continue
                AUC[m+i] = auc

            b = len(users)
            progbar.add(b, values=[('AUC', np.mean(AUC[m:m+b][AUC[m:m+b] >= 0]))])
            m += b

        return np.mean(AUC[AUC>-1])


class EvalCallback(Callback):

    def __init__(self, evaluater: EvalProto, filename: str, env: Environment, loss_var_name='val_loss'):
        super(EvalCallback, self).__init__()
        self.env = env
        self.evaluater = evaluater
        self.loss_var_name = loss_var_name
        outfile = os.path.join(env.get_state()['environment_path'], filename)
        self.outfile = outfile
        if os.path.exists(outfile):
            df = pd.read_csv(outfile)
            self.metric = np.array(df['metric'])
            self.epochs = np.array(df['epoch'])
            self.best = np.max(self.metric)
        else:
            self.best = 0.
            self.metric = np.array([])
            self.epochs = np.array([])

    def on_epoch_end(self, epoch, logs=None):
        metric = self.evaluater.evaluate()
        self.metric = np.append(self.metric, metric)
        self.epochs = np.append(self.epochs, epoch)
        self.env.set_state({self.loss_var_name: metric})

    def on_train_end(self, logs=None):
        self.save_result(self.outfile)

    def save_result(self, outfile):
        pd.DataFrame({'epoch': self.epochs, 'score': self.metric}).to_csv(outfile, index=False)
