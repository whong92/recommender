from .utils import getCatMap, getChunk, procChunk, tf_serialize_example, procSingleRow
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

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

    csvChunks = getChunk('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings.csv',
                         'movieId', 'userId', 'rating', chunksize=10000)

    user_map, item_map = getCatMap(csvChunks)
    user_map_df = pd.DataFrame.from_dict(
        user_map, orient='index', columns=['user_cat']
    )
    item_map_df = pd.DataFrame.from_dict(
        item_map, orient='index', columns=['item_cat']
    )
    print(len(item_map))
    """
    csvChunks = getChunk('D:\\PycharmProjects\\recommender\\data\\ml-latest-small\\ratings.csv',
                         'movieId', 'userId', 'rating', chunksize=50000)

    i = 0

    for (users,items,ratings) in tqdm(generate_tensor_chunk(csvChunks, user_map_df, item_map_df)):

        features_dataset = tf.data.Dataset.from_tensor_slices((users, items, ratings))
        ser_dataset = features_dataset.map(tf_serialize_example)

        filename = 'bla{:03d}.tfrecord'.format(i)
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(ser_dataset)
        i += 1
    """
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