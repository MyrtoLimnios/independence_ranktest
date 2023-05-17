import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.stats import rankdata
from numpy.linalg import norm

rds = 4
random.seed(rds)

def sftplus(x,beta = 1e2):
    return (1. /beta) * np.log(1. + np.exp(beta*x))

def sig(x):
    return 1. / (1. + np.exp(-x))

def scoring_RTB(rank_x, N, u0):
    return - (sftplus(rank_x / (N + 1) - u0) + u0 * sig(1e2 * (rank_x / (N + 1)-u0)))


def scoring_RTB_(rank_x, N, u0):
    if rank_x / (N + 1.) >= u0: return rank_x / (N + 1.)
    else: return 0.0


def stat_emp(scoring, sX, sY):
    N = len(sX) + len(sY)

    alldata = np.concatenate((sX, sY))
    ranked = rankdata(alldata)
    rankx = ranked[:len(sX)]

    loss = np.sum([scoring(rx, N) for rx in rankx]) / len(sX)
    return loss


def stat_emp_RTB(sX, sY, u0):
    N = len(sX) + len(sY)

    alldata = np.concatenate((sX, sY))
    ranked = rankdata(alldata)
    rankx = ranked[:len(sX)]
    loss = np.sum([scoring_RTB_(rx, N, u0) for rx in rankx]) / len(sX)

    return loss


def ect_var_unit(empmean, n, quant = 1.96):
    ect = quant * np.sqrt(empmean*(1- empmean)/n)
    return ect


def ect_var_unit_vec(empmean, N, quant = 1.96):
    ect = []
    for i in range(len(N)):
        ect.append(quant * np.sqrt(empmean[i]*(1- empmean[i])/int(2*N[i]/3)))
    return ect


def normtest(z, alternative):
    """Common code between all the normality-test functions."""
    if alternative == 'less':
        prob = scipy.stats.distributions.norm.cdf(z)
    elif alternative == 'greater':
        prob = scipy.stats.distributions.norm.sf(z)
    elif alternative == 'two-sided':
        prob = 2 * scipy.stats.distributions.norm.sf(np.abs(z))
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")

    if z.ndim == 0:
        z = z[()]

    return z, prob

def sq_distances(X,Y=None):
    """
    If Y=None, then this computes the distance between X and itself
    """
    assert (X.ndim == 2)
    #if X.ndim == 2:
    #    X = X[:, np.newaxis]
    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert(Y.ndim==2)
        assert(X.shape[1]==Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')

    return sq_dists

def median_bandwidth(Z):
    sq_dists = sq_distances(Z)
    np.fill_diagonal(sq_dists, np.nan)
    sq_dists = np.ravel(sq_dists)
    sq_dists = sq_dists[~np.isnan(sq_dists)]
    median_dist = np.median(np.sqrt(sq_dists))

    return np.sqrt(median_dist / 2.0)  


def gauss_kernel(X, Y=None, sigma=1.0):
    sq_dists = sq_distances(X, Y)
    K = np.exp(-sq_dists / (2 * sigma ** 2))
    return K

def HSIC_stat(X, Y):
    medX = median_bandwidth(X)#[:, np.newaxis])
    medY = median_bandwidth(Y)#[:, np.newaxis])
    kernel = lambda X, Y, band: np.exp(- sq_distances(X,Y)/ (2 * band ** 2))
    K_XX = kernel(X, X, medX)
    K_YY = kernel(Y, Y, medY)

    assert len(K_XX) == len(K_YY)
    N = len(K_XX)
    K_XX_ = K_XX
    K_YY_ = K_YY
    H = np.eye(N) - 1.0 / N
    statistic = np.trace(K_XX_.dot(H).dot(K_YY_.dot(H))) / (N ** 2)

    return statistic


def cov_center(X,Y):
    akl = cdist(X, Y, 'sqeuclidean')

    ak = np.asarray([np.sum(akl[i])/ len(X)  for i in range(len(akl))] ) # sum per line
    al = np.asarray( [np.sum(akl.T[i])/ len(X)  for i in range(len(akl.T))])  # sum per column
    cov = akl - ak - al + np.sum(ak) / (len(X) ** 2)

    return cov

def dcor(X,Y):
    assert X.ndim == Y.ndim

    VX = cov_center(X, X)  #/ len(X)**2
    VY = cov_center(Y, Y) #/ len(Y)**2
    VXY = np.sum(np.multiply(VX, VY))  #/ (len(X)**2)
    statistic = VXY / (norm(VX)*norm(VY))

    return statistic

def permutation_test_pval(statistic_fct, X, Y, num_permutations):
    assert X.ndim == Y.ndim

    statistics = np.zeros(num_permutations)
    my_test_raw = statistic_fct(X, Y)

    pval = 1.
    for i in range(num_permutations-1):
        XY = np.vstack((X, Y))
        perm_idx = np.random.permutation(len(XY))
        XY = XY[perm_idx]
        X_ = XY[:len(X)]
        Y_ = XY[len(X):]
        my_test_perm = statistic_fct(X_, Y_)
        statistics[i] = my_test_perm
        if my_test_perm >= my_test_raw:
            pval += 1.

    pval = (1. /num_permutations) * pval

    return pval, statistics














