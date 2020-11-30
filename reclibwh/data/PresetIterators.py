from reclibwh.data.iterators import *
import scipy.sparse as sps
import pandas as pd
import tensorflow as tf
import time

def AUC_data_iter_preset(U: sps.csr_matrix, batchsize=10, rows=None):
    if rows is None: rows = np.arange(U.shape[0])
    return SparseMatRowIterator(batchsize, padded=True, negative=False)({'S': U, 'pad_val': -1, 'rows': rows})

def ALS_data_iter_preset(U: sps.csr_matrix, batchsize=20, rows=None):
    return link([
        {'S': U, 'pad_val': U.shape[1], 'rows': rows},
        SparseMatRowIterator(batchsize, padded=True, negative=False),
        Rename({'rows': 'u_in', 'cols': 'i_in', 'val': 'rhat'}),
        XyDataIterator(ykey='rhat')
    ])

def LMF_data_iter_preset(U: sps.csr_matrix, batchsize=20):
    return link([
        {'S': U, 'pad_val': -1.},
        SparseMatRowIterator(batchsize, padded=False, negative=True),
        Rename({'rows': 'u_in', 'cols': 'i_in', 'val': 'rhat'}),
        XyDataIterator(ykey='rhat')
    ])

def MF_data_iter_preset(df: pd.DataFrame, rnorm=None, batchsize=200):

    return link([
        [df],
        BasicDFDataIter(batchsize),
        Normalizer({} if not rnorm else {'rating': rnorm}),
        Rename({'user': 'u_in', 'item': 'i_in', 'rating': 'rhat'}),
        XyDataIterator(ykey='rhat')
    ])

def MFAsym_data_iter_preset(df: pd.DataFrame, U: sps.csr_matrix, batchsize=200, rnorm=None, Bi: np.array=None, remove_rated_items=True):

    it = BasicDFDataIter(batchsize)
    add_rated_items = AddRatedItems(U, remove_rated_items=remove_rated_items)
    add_bias = AddBias(U, item_key='user_rated_items', pad_val=-1, Bi=Bi)

    def add_1_to_user_rated_items_fn(d):
        d['user_rated_items'] += 1
        return d

    add_1_to_user_rated_items = LambdaDictIterable(add_1_to_user_rated_items_fn)
    rename = Rename({
        'user': 'u_in',
        'item': 'i_in',
        'rating': 'rhat',
        'user_rated_items': 'uj_in',
        'user_rated_ratings': 'ruj_in',
        'bias': 'bj_in',
    })
    nit = Normalizer({} if not rnorm else {'rhat': rnorm, 'ruj_in': rnorm, 'bj_in': rnorm})
    mfit = XyDataIterator(ykey='rhat')

    return link([
        [df], it, add_rated_items, add_bias, add_1_to_user_rated_items, rename, nit, mfit
    ])

def split_df_random(df: pd.DataFrame, num_split=5):

    df = df.iloc[np.random.permutation(len(df))]
    print("splitting df into {:d} random shards".format(num_split))
    L = len(df)
    delta = int(np.round(L / num_split))
    for i in range(num_split):
        start = min(i * delta, L-1)
        end = min((i+1) * delta, L)
        df_split = df.iloc[start:end]
        keys = ['rating', 'item', 'user']
        yield tuple([np.array(df_split.loc[:, k]) for k in keys])

def MFAsym_data_tf_dataset(df, U, rnorm, batchsize, num_workers=4, buffer_size=32):
    """
    Adds a TF Dataset interleave iterator over MFAsym_data_iter_preset, to parallelize and speed up extracting
    data from the dataframe
    :param df:
    :param U:
    :param rnorm:
    :param batchsize:
    :param num_workers:
    :return:
    """

    class MFAsymIterator(tf.data.Dataset):
        """
        generator to consume a dataframe shard, and then iterate over it
        """

        def _generator(a, b, c):

            df = {'rating': a, 'item': b, 'user': c}
            df = pd.DataFrame(df)  # re-constitute into dataframe

            data_train = MFAsym_data_iter_preset(df, U, rnorm=rnorm, batchsize=batchsize)
            for d in data_train:
                yield d

        def __new__(cls, a, b, c):

            return tf.data.Dataset.from_generator(
                cls._generator,
                output_types=(
                    {
                        'u_in': tf.int64, 'i_in': tf.int64, 'uj_in': tf.int64, 'ruj_in': tf.float32,
                        'bj_in': tf.float32
                    },
                    {'rhat': tf.float32}
                ),
                args=(a, b, c,)
            )

    dataset = tf.data.Dataset.\
        from_generator(lambda: split_df_random(df, num_workers), output_types=(tf.float32, tf.int32, tf.int32)).\
        interleave(MFAsymIterator, cycle_length=num_workers, num_parallel_calls=num_workers, block_length=1).\
        prefetch(buffer_size)

    class TFDatasetWrapper:

        def __iter__(self):
            for d in dataset: yield d

        def __len__(self):
            df_shards = split_df_random(df, num_workers)
            l = 0
            for df_shard in df_shards:
                l += int(np.ceil(len(df_shard[0]) / batchsize))
            return l

    return TFDatasetWrapper()

if __name__=="__main__":

    from tqdm import tqdm

    data_folder = "data/ml-latest-small"
    d = ExplicitDataFromCSV(True, data_folder=data_folder)
    df_train, df_test = d.make_training_datasets(dtype='df')
    Utrain, _ = d.make_training_datasets(dtype='sparse')
    rnorm = {'loc': 0.0, 'scale': 5.0}

    for d in tqdm(MFAsym_data_tf_dataset(df_train, Utrain, rnorm=rnorm, batchsize=2000)):
        time.sleep(0.5)
        pass
