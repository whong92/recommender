from .utils import getCatMap, getChunk, procChunk, tf_serialize_example, procSingleRow
import pandas as pd
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Pool
from queue import Queue
from pyspark import SparkContext as sc

tf.logging.set_verbosity(tf.logging.INFO)

def generate_tensor_chunk(chunks, user_map_df, item_map_df):
    for chunk in chunks:
        chunk = procChunk(chunk, user_map_df, item_map_df)
        yield chunk
        """
        yield (np.array(chunk['user'], dtype=np.int32),
               np.array(chunk['item'], dtype=np.int32),
               np.array(chunk['rating'], dtype=np.float64)
               )
        """

def generate_single_row(rows, user_map, item_map):
    for row in rows:
        chunk = procSingleRow(row, user_map, item_map)
        yield (np.int32(np.squeeze(chunk.iloc[0]['user'])),
               np.int32(np.squeeze(chunk.iloc[0]['item'])),
               np.int32(np.squeeze(chunk.iloc[0]['rating']))
               )

def serialize_and_write_to_file(dataset, filename):
    print("writing dataset to  {:s}".format(filename))
    tf.logging.set_verbosity(tf.logging.DEBUG)
    ser_dataset = dataset.map(tf_serialize_example, num_parallel_calls=4)
    writer = tf.data.experimental.TFRecordWriter(filename)
    sess = tf.Session()
    sess.run(writer.write(ser_dataset))

def serialize_df_and_write_to_file(df, filename):
    # tf.logging.set_verbosity(tf.logging.DEBUG)
    print(" ============= serializing dataset to  {:s} ============= ".format(filename))
    ds = tf.data.Dataset.from_tensor_slices((df['user'], df['item'], df['rating']))
    ser_dataset = ds.map(tf_serialize_example, num_parallel_calls=2)
    print(" ============= writing dataset to  {:s} ============= ".format(filename))
    writer = tf.data.experimental.TFRecordWriter(filename)
    sess = tf.Session()
    sess.run(writer.write(ser_dataset))
    print(" ============= finished writing dataset to  {:s} ============= ".format(filename))

def main():

    src_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-20m\\ratings.csv'
    dst_folder = 'D:\\PycharmProjects\\recommender\\data\\tmp-20m'

    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    chunksize = 2500000

    """
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
    """

    user_map_df = pd.DataFrame.from_csv(os.path.join(dst_folder, 'user_map_df.csv'))
    item_map_df = pd.DataFrame.from_csv(os.path.join(dst_folder, 'item_map_df.csv'))
    jobQ = []

    csvChunks = getChunk(src_csv, 'movieId', 'userId', 'rating', chunksize=chunksize)

    i = 0
    j = 0

    active_train_df = pd.DataFrame({
        'user': np.array([], dtype=np.int32),
        'item': np.array([], dtype=np.int32),
        'rating': np.array([], dtype=np.float64),
    })
    active_test_df = pd.DataFrame({
        'user': np.array([], dtype=np.int32),
        'item': np.array([], dtype=np.int32),
        'rating': np.array([], dtype=np.float64),
    })

    p = Pool(3)

    for chunk in tqdm(generate_tensor_chunk(csvChunks, user_map_df, item_map_df)):

        cur_len = len(chunk)
        test_offset = int(0.8 * cur_len)
        #features_dataset = tf.data.Dataset.from_tensor_slices((users, items, ratings))
        for mode in ['train', 'test']:

            if mode is 'train' and test_offset == 0:
                continue
            if mode is 'test' and test_offset == cur_len:
                continue

            if mode is 'train':
                #fetched_train_dataset = features_dataset.take(test_offset)
                #active_train_dataset = active_train_dataset.concatenate(fetched_train_dataset)
                active_train_df = pd.concat([active_train_df, chunk.iloc[:test_offset]])
                #active_train_size += test_offset
                # if active_train_size > chunksize:
                #     filename = os.path.join(dst_folder, 'bla_train{:03d}.tfrecord'.format(i))
                #     p.map_async(serialize_and_write_to_file(active_train_dataset, filename))
                #     active_train_dataset = tf.data.Dataset.from_tensor_slices(
                #         (np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)))
                #     active_train_size = 0
                #     i += 1
                if len(active_train_df) > chunksize:
                    filename = os.path.join(dst_folder, 'bla_train{:03d}.tfrecord'.format(i))
                    if len(jobQ) >= 3:
                        job = jobQ.pop(0)
                        job.get()
                    job = p.apply_async(serialize_df_and_write_to_file, (active_train_df.copy(), filename))
                    jobQ.append(job)
                    active_train_df = pd.DataFrame({
                        'user': np.array([], dtype=np.int32),
                        'item': np.array([], dtype=np.int32),
                        'rating': np.array([], dtype=np.float64),
                    })
                    i += 1
            else:
                #fetched_test_dataset = features_dataset.skip(test_offset)
                #fetched_test_dataset = fetched_test_dataset.take(len(users) - test_offset)
                #active_test_dataset = active_test_dataset.concatenate(fetched_test_dataset)
                active_test_df = pd.concat([active_test_df, chunk.iloc[test_offset:]])
                #active_test_size += len(users) - test_offset
                # if active_test_size > chunksize:
                #     filename = os.path.join(dst_folder, 'bla_test{:03d}.tfrecord'.format(j))
                #     p.map_async(serialize_and_write_to_file(active_test_dataset, filename))
                #     active_test_dataset = tf.data.Dataset.from_tensor_slices(
                #         (np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)))
                #     active_test_size = 0
                #     j += 1
                if len(active_test_df) > chunksize:
                    filename = os.path.join(dst_folder, 'bla_test{:03d}.tfrecord'.format(j))
                    if len(jobQ) >= 3:
                        job = jobQ.pop(0)
                        job.get()
                    job = p.apply_async(serialize_df_and_write_to_file, (active_test_df.copy(), filename))
                    jobQ.append(job)
                    active_test_df = pd.DataFrame({
                        'user': np.array([], dtype=np.int32),
                        'item': np.array([], dtype=np.int32),
                        'rating': np.array([], dtype=np.float64),
                    })
                    j += 1

    # write final chunk
    filename = os.path.join(dst_folder, 'bla_train{:03d}.tfrecord'.format(i))
    p.apply_async(serialize_df_and_write_to_file, (active_train_df, filename))

    filename = os.path.join(dst_folder, 'bla_test{:03d}.tfrecord'.format(j))
    p.apply_async(serialize_df_and_write_to_file, (active_test_df, filename))

    p.close()
    p.join()

def main_spark():

    src_csv = 'D:\\PycharmProjects\\recommender\\data\\ml-20m\\ratings.csv'
    dst_folder = 'D:\\PycharmProjects\\recommender\\data\\tmp-20m'

    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    numPartitions = 5

    from pyspark.sql import SparkSession

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    user_map_df = spark.read.format("csv").option("header", "true").load(os.path.join(dst_folder, 'user_map_df.csv'))
    item_map_df = spark.read.format("csv").option("header", "true").load(os.path.join(dst_folder, 'item_map_df.csv'))
    ratings_df = spark.read.format("csv").option("header", "true").load(src_csv)

    U = user_map_df.alias("U")
    I = item_map_df.alias("I")
    R = ratings_df.alias("R")

    all_dataset = R.join(U, R['userId'] == U['idx'], how='left').join(I, R['movieId'] == I['idx'], how='left').\
        select("user_cat", "item_cat", "rating")

    all_dataset_split = all_dataset.repartition(numPartitions)

    def to_tf_record(splitIndex, glom):
        print("converting spark df to tf record")
        users = None
        items = None
        ratings = None
        for rows in glom:
            users = np.zeros(shape=(len(rows)), dtype=np.int64)
            items = np.zeros(shape=(len(rows)), dtype=np.int64)
            ratings = np.zeros(shape=(len(rows)), dtype=np.float64)
            for r, row in enumerate(rows):
                users[r] = row.user_cat
                items[r] = row.item_cat
                ratings[r] = row.rating
        print("finished collecting")
        filename = os.path.join(dst_folder, 'bla_train{:03d}.tfrecord'.format(splitIndex))
        features_dataset = tf.data.Dataset.from_tensor_slices((users, items, ratings))
        print("writing to tfrecords")
        serialize_and_write_to_file(features_dataset, filename)
        return [filename]

    print(all_dataset_split.rdd.glom().mapPartitionsWithIndex(to_tf_record).collect())

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


if __name__=="__main__":

    #tf.enable_eager_execution()
    main()

    #import cProfile, pstats

    # pr = cProfile.Profile()
    # pr.enable()
    # pr.run('main_spark()')
    # pr.disable()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr).sort_stats(sortby)
    # ps.print_stats(50)