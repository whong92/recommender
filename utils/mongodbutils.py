import pymongo as pm

def getDB(dbconn, db):
    myclient = pm.MongoClient(dbconn)
    return myclient[db]

def getClient(dbconn):
    return pm.MongoClient(dbconn)
