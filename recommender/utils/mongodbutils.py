import pymongo as pm
import numpy as np

def features_json(ids, features):
    return [
        {"_id":int(i), "feature":list(f)} for i,f in zip(ids, features)
    ]

def attempt_bulk_write(table, query):
    try:
        result = table.bulk_write(query)
    except pm.errors.BulkWriteError as bwe:
        raise Exception("Error doing build write: {}".format(bwe.details))
    return result.acknowledged

def attempt_insert_many(table, docs):
    try:
        result = table.insert_many(docs)
    except pm.errors.PyMongoError as pe:
        raise Exception("Pymongo error from insertion operation: {}".format(pe.details))
    return result.acknowledged

class DataService:

    def __init__(self, conn, db_name):
         self.conn = conn
         self.db_name = db_name
         self.dbclient = pm.MongoClient(conn)
         self.db = self.dbclient[db_name]

    def insert_features(self, table, features):
        assert len(features) > 0, 'no features to insert'
        feat_table = self.db["features/{:s}".format(table)]
        return attempt_insert_many(feat_table, [{'_id': int(f), 'feature': list(features[f])} for f in features])

    def update_band(self, table, band_num, bucket):

        assert len(bucket) > 0, 'no buckets to insert'

        query = [
            pm.UpdateOne(
                {'_id': int(h)},
                {'$push': {'vals': {'$each': list(bucket[h])}}},
                upsert=True
            )
            for h in bucket
        ]
        band_table = self.db['{:s}/band{:03d}'.format(table, band_num)]
        return attempt_bulk_write(band_table, query)

    def get_bucket(self, table, band_num, h):
        band = self.db['{:s}/band{:03d}'.format(table, band_num)]
        result = []
        bucket_cursor = band.find({'_id': int(h)})
        for bucket in bucket_cursor:
            result.extend(bucket['vals'])
        return result

    def get_features_by_ids(self, table, ids=None):
        feat_table = self.db["features/{:s}".format(table)]
        if ids is not None:
            results = feat_table.find({'_id':{'$in':ids}})
        else:
            results = feat_table.find()
        features = []
        for result in results:
            features.append(result)
        return features

    def insert_movie_mds(self, table, titles, ids, embedding_ids, r_count):
        md_table = self.db['{:s}/movie_metadata'.format(table)]
        return attempt_insert_many(
            md_table,
            [{'_id': int(i), 'title':t, 'embedding_id':int(e), 'r_count':r}
             for t,i,e,r in zip(titles, ids,embedding_ids, r_count)]
        )

    def get_movie_mds_by_embedding_ids(self, table, embedding_ids=None):
        md_table = self.db['{:s}/movie_metadata'.format(table)]
        if embedding_ids is not None:
            results = md_table.find({'_id':{'$in':embedding_ids}})
        else:
            results = md_table.find()
        features = []
        for result in results:
            features.append(result)
        return features

    def drop_features(self, table):
        self.db["features/{:s}".format(table)].drop()

    def drop_lsh(self, table, band_num):
        self.db['{:s}/band{:03d}'.format(table, band_num)].drop()