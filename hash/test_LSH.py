from .LSH import LSH
from .Signature import MakeSignature
import numpy as np
import scipy.sparse as sps
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os

def generate_random_vectors(M, N):
    # reference vector
    ref = np.random.normal(size=(M,))
    ref /= np.linalg.norm(ref)

    # random other vectors
    ref2 = np.random.normal(size=(M, N))
    C = np.cross(ref, ref2, axisb=0)
    C = np.divide(C, np.expand_dims(np.linalg.norm(C, axis=1), axis=1))
    C = np.multiply(C, np.random.uniform(0, np.pi, size=(N, 1)))
    rot = R.from_rotvec(C)
    X = sps.csc_matrix(rot.apply(ref).transpose())

    return ref, X

def generate_lsh_and_insert(ref, X, M, N, Hpp, B):

    csh = MakeSignature('CosineHash', num_row=3, num_hpp=Hpp)
    lsh = LSH(csh, num_bands=B)
    lsh.insert(X)

    sim_set = lsh.find_similar(sps.csc_matrix(np.expand_dims(ref, axis=1), shape=(M, 1)))

    return lsh, csh, sim_set

if __name__=="__main__":

    # test saving and re-loading
    N = 20
    M = 3
    Hpp = 10
    B = 5

    ref, X = generate_random_vectors(M, N)
    lsh, csh, sim_set = generate_lsh_and_insert(ref, X, M, N, Hpp, B)

    # save initial lsh
    lsh.save('./bla.json')
    # reload lsh and expect to get same sim_set
    lsh2 = LSH(csh, num_bands=B, path='./bla.json')
    sim_set_2 = lsh2.find_similar(sps.csc_matrix(np.expand_dims(ref, axis=1), shape=(M,1)))
    assert len(sim_set - sim_set_2)==0

    os.remove('./bla.json')

    # are we profiling?
    profile = False

    def test_cosine_hash():

        from scipy.spatial.transform import Rotation as R

        N = 5000
        M = 3
        Hpp = 300
        NumBands = np.array([5, 10, 20, 30, 50, 100])

        ref, X = generate_random_vectors(M, N)

        for b, B in enumerate(NumBands):

            lsh, csh, sim_set = generate_lsh_and_insert(ref, X, M, N, Hpp, B)
            Y = np.arccos(ref * X[:, list(sim_set)])

            if not profile:
                plt.subplot(len(NumBands),1, b+1)
                plt.hist(Y, bins=np.linspace(0, np.pi, 50))

        if not profile:
            plt.show()


    if not profile:
        test_cosine_hash()
    else:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
        pr.run('test_cosine_hash()')
        pr.disable()
        sortby = 'cumulative'
        ps = pstats.Stats(pr).sort_stats(sortby)
        ps.print_stats()
