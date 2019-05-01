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
        raise Exception("Error doing build write: {:s}".format(bwe.details))
    return result.acknowledged

class DataService:

    def __init__(self, conn, db):
         self.dbclient = pm.MongoClient(conn)
         self.db = self.dbclient[db]

    def insert_features(self, table, features):

        assert len(features) > 0, 'no features to insert'

        feat_table = self.db["features/{:s}".format(table)]
        query = [
            pm.InsertOne(
                {'_id': int(f), 'feature': list(features[f])},
            )
            for f in features
        ]

        return attempt_bulk_write(feat_table, query)

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