import pymongo as pm
from .Signature import MakeSignature
import scipy.sparse as sps
from .LSH import LSHDB
from .test_LSH import generate_random_vectors, generate_lsh_and_insert
from ..utils.mongodbutils import DataService
import numpy as np
import matplotlib.pyplot as plt

def generate_lshdb_and_insert(url, db_name, ref, X, M, N, Hpp, B):

    csh = MakeSignature('CosineHash', num_row=3, num_hpp=Hpp)
    ds = DataService(url, db_name)
    lsh = LSHDB(ds, 'hullabalooza', csh, B)
    lsh.insert(X, flush_every=5000)

    sim_set = lsh.find_similar(np.expand_dims(ref, axis=1))

    return lsh, csh, sim_set

if __name__=="__main__":

    dbconn = ("localhost", 27017)
    db_name = "test"
    url = "mongodb://{:s}:{:d}/".format(*dbconn)

    def test_vs_dict_lsh():

        N = 20
        M = 3
        Hpp = 10
        B = 5

        # X-check results with dict-based LSH
        ref, X = generate_random_vectors(M, N)
        lsh, csh, sim_set = generate_lsh_and_insert(ref, X, M, N, Hpp, B)
        # save initial lsh
        lsh.save('./bla.json')

        client = pm.MongoClient(*dbconn)
        db = client[db_name]

        ds = DataService(url, db_name)
        lshdb = LSHDB(ds, 'hullabalooza', csh, B)
        lshdb.insert(X)
        sim_set_db = lshdb.find_similar(np.expand_dims(ref, axis=1))
        assert len(sim_set - sim_set_db) == 0

        # clean up
        client.drop_database(db)


    def test_cosine_hash():

        N = 5000
        M = 3
        Hpp = 300
        NumBands = np.array([5, 10, 20, 30, 50, 100])

        ref, X = generate_random_vectors(M, N)

        for b, B in enumerate(NumBands):

            client = pm.MongoClient(*dbconn)
            db = client[db_name]
            lsh, csh, sim_set = generate_lshdb_and_insert(url, db_name, ref, X, M, N, Hpp, B)
            # clean up
            client.drop_database(db)

            Y = np.arccos(np.matmul(ref, X[:, list(sim_set)]))
            plt.subplot(len(NumBands),1, b+1)
            plt.hist(Y, bins=np.linspace(0, np.pi, 50))

        plt.show()

    for test in [test_vs_dict_lsh, test_cosine_hash]:

        test()