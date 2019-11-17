import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class ALS:

    def __init__(self, mode='train', N=100, M=100, K=10, lamb=1e-06, alpha=40., model_path=None):
        self.R = None #U # utility |user|x|items|, sparse, row major
        self.mode = mode
        self.model_path = model_path
        self.M = M
        self.N = N
        if mode == 'train':
            self.K = K
            # dense feature matrices
            self.X = np.random.normal(0, 1/np.sqrt(self.K), size=(N, self.K))
            self.Y = np.random.normal(0, 1/np.sqrt(self.K), size=(M, self.K))
            self.lamb = lamb
            self.alpha = alpha
        else:
            assert model_path is not None, "model path required in predict mode"
            self.X = np.load(os.path.join(model_path, 'X.npy'))
            self.Y = np.load(os.path.join(model_path, 'Y.npy'))

    def _run_single_step(self, Y, X, C, R, p_float):

        assert self.mode == 'train', "cannot call when mode is not train"

        lamb = self.lamb
        Y2 = np.tensordot(Y.T, Y, axes=1)
        Lamb = lamb*np.eye(self.K)
        alpha = self.alpha
        Xp = X.copy()

        for u, x in enumerate(X):
            cu = C[u, :]
            ru = R[u, :]
            p_idx = sps.find(ru)[1]
            Yp = Y[p_idx, :]
            rp = ru[:, p_idx].toarray()[0]
            L = np.linalg.inv(Y2 + np.matmul(Yp.T, alpha*np.multiply(np.expand_dims(rp, axis=1), Yp)) + Lamb)
            cup = cu[:, p_idx].toarray()[0]
            p = p_float[u, :]
            pp = p[:, p_idx].toarray()[0]
            xp = np.matmul(L, np.tensordot(Yp.T, np.multiply(cup, pp), axes=1))
            Xp[u, :] = xp

        return Xp

    def _run_single_step_naive(self, Y, X, _R, p_float):

        assert self.mode == 'train', "cannot call when mode is not train"

        lamb = self.lamb
        Lamb = lamb * np.eye(self.K)
        alpha = self.alpha

        C = _R.copy().toarray()
        C = 1 + alpha*C
        P = p_float.copy().toarray()

        Xp = X.copy()

        for u, x in enumerate(X):
            cu = C[u, :]
            Cu = np.diag(cu)
            pu = P[u,:]
            L = np.linalg.inv(np.matmul(Y.T, np.matmul(Cu, Y)) + Lamb)
            xp = np.matmul(L, np.matmul(Y.T,np.matmul(Cu, pu)))
            Xp[u, :] = xp

        return Xp

    def _calc_loss(self, Y, X, _C, _R, _p):

        assert self.mode == 'train', "cannot call when mode is not train"

        lamb = self.lamb
        p = _p.copy().toarray()
        R = _R.copy().toarray()
        C = _C.copy().toarray()
        loss = np.sum(np.multiply(C, np.square(p - np.matmul(X, Y.T))))
        loss += lamb*(np.mean(np.linalg.norm(X, 2, axis=1)) + np.mean(np.linalg.norm(Y, 2, axis=1)))
        return loss

    def train(self, U):

        assert self.mode == 'train', "cannot call when mode is not train"
        R = U
        p = R.astype(np.bool, copy=True).astype(np.float, copy=True)
        C = R.copy()
        C.data = 1 + self.alpha*C.data

        for i in tqdm(range(self.K)):
            Xp = self._run_single_step(self.Y, self.X, C, R, p)
            self.X = Xp
            Yp = self._run_single_step(self.X, self.Y, C.T, R.T, p.T)
            self.Y = Yp

    def save(self, model_path=None):

        if model_path is not None:
            self.model_path = model_path
        assert self.model_path is not None, "model path not specified"

        np.save(os.path.join(self.model_path, "X.npy"), self.X)
        np.save(os.path.join(self.model_path, "Y.npy"), self.Y)



if __name__=="__main__":

    M = 100
    N = 50
    P = 150
    np.random.seed(42)
    data = np.random.randint(0, 10, size=(P,))
    c = np.random.randint(0, M*N, size=(P,))
    rows = c//M
    cols = c%M

    als = ALS(N=N, M=M)

    U = sps.csr_matrix((data, (rows, cols)), shape=(N,M))
    R = U
    p = R.copy().astype(np.bool).astype(np.float, copy=True)
    C = R.copy()
    C.data = 1 + als.alpha * C.data

    Xp = als._run_single_step_naive(als.Y, als.X, R, p)
    print(Xp[1,:])

    steps = 20
    losses = np.zeros(shape=(steps*2))
    for s in range(steps):
        Xp = als._run_single_step(als.Y, als.X, C, R, p)
        als.X = Xp
        losses[2*s] = als._calc_loss(als.Y, als.X, C, R, p)
        print(losses[2*s])
        Yp = als._run_single_step(als.X, als.Y, C.T, R.T, p.T)
        als.Y = Yp
        losses[2*s+1] = als._calc_loss(als.Y, als.X, C, R, p)
        print(losses[2*s+1])

    model_path = "D:\\PycharmProjects\\recommender\\models\\ALS_toy_2019_11_18"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    als.save(model_path)

    plt.plot(np.log(losses), 'o:')
    plt.show()