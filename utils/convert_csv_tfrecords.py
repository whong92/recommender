from .utils import getCatMap, getChunk, procChunk, tf_serialize_example, procSingleRow
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os

def generate_tensor_chunk(chunks, user_map_df, item_map_df):
    for chunk in chunks:
        chunk = procChunk(chunk, user_map_df, item_map_df)
        yield (np.array(chunk['user'], dtype=np.int32),
               np.array(chunk['item'], dtype=np.int32),
               np.array(chunk['rating'], dtype=np.float64)
               )

def generate_single_row(rows, user_map, item_map):
    for row in rows:
        chunk = procSingleRow(row, user_map, item_map)
        yield (np.int32(np.squeeze(chunk.iloc[0]['user'])),
               np.int32(np.squeeze(chunk.iloc[0]['item'])),
               np.int32(np.squeeze(chunk.iloc[0]['rating']))
               )

if __name__=="__main__":
    tf.enable_eager_execution()
    src_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings.csv'
    dst_folder = 'D:\\PycharmProjects\\recommender\\data\\tmp'

    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder )

    chunksize = 200000

    csvChunks = getChunk(src_csv, 'movieId', 'userId', 'rating', chunksize=chunksize)

    user_map, item_map, ds_len = getCatMap(csvChunks)
    user_map_df = pd.DataFrame.from_dict(
        user_map, orient='index', columns=['user_cat']
    )
    item_map_df = pd.DataFrame.from_dict(
        item_map, orient='index', columns=['item_cat']
    )
    user_map_df.to_csv(os.path.join(dst_folder, 'user_map_df.csv'))
    item_map_df.to_csv(os.path.join(dst_folder, 'item_map_df.csv'))

    train_len = int(0.8 * ds_len)
    test_len = int(0.2 * ds_len)

    csvChunks = getChunk(src_csv, 'movieId', 'userId', 'rating', chunksize=chunksize)

    i = 0
    j = 0
    train_len_so_far = 0

    for (users,items,ratings) in tqdm(generate_tensor_chunk(csvChunks, user_map_df, item_map_df)):

        cur_len = len(users)
        test_offset = int(0.8*cur_len)
        features_dataset = tf.data.Dataset.from_tensor_slices((users, items, ratings)).shuffle(cur_len)

        for mode in ['train', 'test']:

            if mode is 'train' and test_offset==0:
                continue
            if mode is 'test' and test_offset==cur_len:
                continue

            if mode is 'train':
                active_dataset = features_dataset.take(test_offset)
                filename = os.path.join(dst_folder, 'bla_train{:03d}.tfrecord'.format(i))
                i += 1
            else:
                active_dataset = features_dataset.skip(test_offset)
                active_dataset = active_dataset.take(len(users) - test_offset)
                filename = os.path.join(dst_folder, 'bla_test{:03d}.tfrecord'.format(j))
                j += 1

            ser_dataset = active_dataset.map(tf_serialize_example)
            writer = tf.data.experimental.TFRecordWriter(filename)
            writer.write(ser_dataset)

    """
    filenames = ['D:\\PycharmProjects\\recommender\\bla{:03d}.tfrecord'.format(i) for i in range(3)]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    feature_description = {
        'user': tf.FixedLenFeature([], tf.int64, default_value=0),
        'item': tf.FixedLenFeature([], tf.int64, default_value=0),
        'rating': tf.FixedLenFeature([], tf.float32, default_value=0),
    }
    raw_dataset = raw_dataset.shuffle(10000)
    raw_dataset = raw_dataset.repeat(10).batch(32)
    parsed_dataset = raw_dataset.map(lambda x: tf.parse_example(x, feature_description))
    print(parsed_dataset)
    for record in parsed_dataset.take(10):
        print(repr(record))
    """