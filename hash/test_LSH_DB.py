import pymongo as pm
import scipy.sparse as sps
from .LSH import LSHDB
from .test_LSH import generate_random_vectors, generate_lsh_and_insert
import numpy as np

if __name__=="__main__":

    dbconn = ("localhost", 27017)
    db_name = "test"
    url = "mongodb://{:s}:{:d}/".format(*dbconn)
    client = pm.MongoClient(*dbconn)
    db = client[db_name]

    N = 20
    M = 3
    Hpp = 10
    B = 5

    # X-check results with dict-based LSH
    ref, X = generate_random_vectors(M, N)
    lsh, csh, sim_set = generate_lsh_and_insert(ref, X, M, N, Hpp, B)
    # save initial lsh
    lsh.save('./bla.json')

    lshdb = LSHDB(csh, B, url, db_name)
    lshdb.insert(X)
    sim_set_db = lshdb.find_similar(sps.csc_matrix(np.expand_dims(ref, axis=1), shape=(M,1)))
    assert len(sim_set - sim_set_db)==0

    print(sim_set, sim_set_db)

    # clean up
    client.drop_database(db)
