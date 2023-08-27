import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

from memory_profiler import profile 
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

# @profile
@timing
def test(pairwise_dists, verbose=True):
    N = pairwise_dists.shape[0]
    sigmas = cp.Variable(N)

    constraints = [sigmas >= 1e-8]
    for i in range(N-1):
        for j in range(i+1, N):
            constraints.append(sigmas[i] + sigmas[j] >= pairwise_dists[i, j])
    obj = cp.Minimize(cp.sum_squares(sigmas))
    prob = cp.Problem(obj, constraints)
    print("Solving with MOSEK...")
    prob.solve('MOSEK', verbose=verbose)
    print("sigma=", sigmas.value)
    return sigmas.value


@timing
def test_vectorised(pairwise_dists, verbose=True):
    N = pairwise_dists.shape[0]
    sigmas = cp.Variable(N)

    constraints = [sigmas >= 1e-8]
    for i in range(N-1):
        constraints.append(sigmas[i] + sigmas[i+1:] >= pairwise_dists[i, i+1:])
    obj = cp.Minimize(cp.sum_squares(sigmas))
    prob = cp.Problem(obj, constraints)
    print("Solving with MOSEK...")
    prob.solve('MOSEK', verbose=verbose)
    print("sigma=", sigmas.value)
    return sigmas.value


if __name__ == "__main__":

    np.random.seed(3)
    ## Uncomment to solve a small problem
    #Z = np.load('small_noises.npy').reshape(200, -1)
    Z = np.load('noises.npy').reshape(50000, -1)
    weight = 1
    pairwise_dists = weight * pairwise_distances(Z)
    print('Shape of pairwise_dists:', pairwise_dists.shape)
    V = test_vectorised(pairwise_dists)
    
