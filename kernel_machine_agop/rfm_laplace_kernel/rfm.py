import numpy as np
from numpy.linalg import solve, pinv
from sklearn.base import BaseEstimator
import time
from tqdm import tqdm

def euclidean_distances(samples, centers, M=None, squared=True):
    if M is None:
        samples_norm = np.sum(samples**2, axis=1, keepdims=True)
    else:
        samples_norm = (samples @ M)  * samples
        samples_norm = np.sum(samples_norm, axis=1, keepdims=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        if M is None:
            centers_norm = np.sum(centers**2, axis=1, keepdims=True)
        else:
            centers_norm = (centers @ M) * centers
            centers_norm = np.sum(centers_norm, axis=1, keepdims=True)

    centers_norm = np.reshape(centers_norm, (1, -1))
    distances = samples @ (M @ centers.T)
    distances *= -2
    distances = distances + samples_norm + centers_norm
    if not squared:
        distances = np.where(distances < 0, 0, distances)
        distances = np.sqrt(distances)

    return distances

def laplace_kernel(samples, centers, bandwidth, M=None):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, M=M, squared=False)
    kernel_mat = np.where(kernel_mat < 0, 0, kernel_mat)
    gamma = 1. / bandwidth
    kernel_mat *= -gamma
    kernel_mat = np.exp(kernel_mat)
    return kernel_mat


def get_grads(X, sol, L, P, max_num_samples=20000, centering=True):
    indices = np.random.randint(len(X), size=max_num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = laplace_kernel(X, x, L, M=P)

    dist = euclidean_distances(X, x, M=P, squared=False)
    dist = np.where(dist < 1e-10, 0, dist)

    with np.errstate(divide='ignore'):
        K = K/dist

    K[K == float("Inf")] = 0.

    a1 = sol.T
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)

    X1 = (X @ P).reshape(n, 1, d)

    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c*d)

    step2 = K.T @ step1
    del step1

    step2 = step2.reshape(-1, c, d)

    a2 = sol
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1/L
    if centering:
        G_mean = np.expand_dims(np.mean(G, axis=0), axis=0)
        G = G - G_mean
    return G

def egop(G, verbose=False, diag_only=False):
    M = 0.
    chunks = len(G) // 20 + 1
    batches = np.array_split(G, chunks)
    if verbose:
        for i in tqdm(range(len(batches))):
            grad = batches[i]
            gradT = np.swapaxes(grad, 1, 2)
            if diag_only:
                T = np.sum(gradT * gradT, axis=-1)
                M += np.sum(T, axis=0)
            else:
                M += np.sum(gradT @ grad, axis=0)
                del grad, gradT
    else:
        for i in range(len(batches)):
            grad = batches[i]
            gradT = np.swapaxes(grad, 1, 2)
            if diag_only:
                T = np.sum(gradT * gradT, axis=-1)
                M += np.sum(T, axis=0)
            else:
                M += np.sum(gradT @ grad, axis=0)
                del grad, gradT
    M /= len(G)
    if diag_only:
        M = np.diag(M)
    return M

class RFM(BaseEstimator):
    def __init__(self, kernel="laplace"):
        self.kernel=kernel
        self.X_train = None
        self.alphas = None
        self.M = None
        self.L = None
        self.reg = None

    def fit(self, X_train, y_train, reg=1e-3, bandwidth=10, num_iters=5,
            M=None, centering=True, verbose=False, diag_only=False):
        self.X_train = X_train
        n, d = X_train.shape
        if M is None:
            M = np.eye(d)
        self.M = M
        self.L = bandwidth
        self.reg = reg

        for iter_idx in range(num_iters):
            if verbose:
                print("Starting Iteration: " + str(iter_idx))
            start = time.time()
            K_train = laplace_kernel(X_train, X_train, self.L, M=M)
            sol = solve(K_train + reg * np.eye(n), y_train).T
            end = time.time()
            if verbose:
                print("Solved Kernel Regression in " + str(end - start) + " seconds.")
            self.alphas = sol

            start = time.time()
            G = get_grads(X_train, self.alphas, self.L, M, centering=centering)
            end = time.time()
            if verbose:
                print("Computed Gradients in "  + str(end - start) + " seconds.")

            start = time.time()
            M = egop(G, verbose=verbose, diag_only=diag_only)
            end = time.time()
            if verbose:
                print("Computed EGOP in " + str(end - start) + " seconds.")
                print("===============================================================")
            self.M = M

        start = time.time()
        K_train = laplace_kernel(X_train, X_train, self.L, M=M)
        sol = solve(K_train + reg * np.eye(n), y_train).T
        end = time.time()
        if verbose:
            print("Solved Final Kernel Regression in " + str(end - start) + " seconds.")
        self.alphas = sol
        return self

    def predict(self, X_test):
        L = self.L
        M = self.M
        K_test = laplace_kernel(self.X_train, X_test, L, M=M)
        preds = (self.alphas @ K_test).T
        return preds

    def get_alphas(self):
        return self.alphas

    def get_M(self):
        return self.M
