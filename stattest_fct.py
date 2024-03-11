"""
author: Myrto Limnios, myli@math.ku.dk
"""

""" 
Script gathering functions used in the main scripts to test for independence between two datasamples X and Y, 
 used in the paper Section Numerical Experiments (Sec. 4):
  "On Ranking-based Tests of Independence"

Useful functions for the implementation of ranking-based independence test, and for SoA methods, namely 
HSIC [Gretton et al. 2007], and dCor with L1 and L2 norms [Szekely et al. 2004]
"""




import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.stats import rankdata
from numpy.linalg import norm
from scipy.stats import distributions

rds = 4
random.seed(rds)

def sftplus(x,beta = 1e2):
    return (1. /beta) * np.log(1. + np.exp(beta*x))

def sig(x):
    return 1. / (1. + np.exp(-x))

def scoring_RTB(rank_x, N, u0):
    """
    :param rank_x: vector of ranks for sample X
    :param N: total sample size
    :param u0: proportion of best ranked instances
    :return: Returns the differentiable version of the score-generating function RTB
    """
    return - (sftplus(rank_x / (N + 1) - u0) + u0 * sig(1e2 * (rank_x / (N + 1)-u0)))


def scoring_RTB_(rank_x, N, u0):
    """
     :param rank_x: vector of ranks for sample X
     :param N: total sample size
     :param u0: proportion of best ranked instances
     :return: Returns the score-generating function RTB
     """
    if rank_x / (N + 1.) >= u0: return rank_x / (N + 1.)
    else: return 0.0


def stat_emp(scoring, sX, sY):
    """
    :param scoring: score-generating function
    :param sX: scored sample X using \widehat{s}
    :param sY: scored sample Y using \widehat{s}
    :return: Returns the W_phi statistic based on the scored (univariate) samples sX and sY
    """
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
    """
    Common code for estimating the p-value with Standard Gaussian distribution
    """
    if alternative == 'less':
        prob = distributions.norm.cdf(z)
    elif alternative == 'greater':
        prob = distributions.norm.sf(z)
    elif alternative == 'two-sided':
        prob = 2 * distributions.norm.sf(np.abs(z))

    return z, prob

def sq_distances(X,Y=None, norm = 2.):
    """
    Computes the distance between X and itself or Y, using the p-norm, by default norm=2.
    """
    assert (X.ndim == 2)
    if X.ndim == 2:  X = X[:, np.newaxis]
    if Y is None:
        sq_dists = squareform(pdist(X, 'minkowski', p=norm))
    else:
        if Y.ndim == 2:  Y= Y[:, np.newaxis]
        sq_dists = cdist(X, Y, 'minkowski', p=norm)

    return sq_dists

def median_bandwidth(Z):
    """
    Takes Z data sample and returns the median of the in-sample Eucledian distances
    """
    sq_dists = sq_distances(Z)
    np.fill_diagonal(sq_dists, np.nan)
    sq_dists = np.ravel(sq_dists)
    sq_dists = sq_dists[~np.isnan(sq_dists)]
    median_dist = np.median(np.sqrt(sq_dists))

    return np.sqrt(median_dist / 2.0)  # our kernel uses a bandwidth of 2*(sigma**2)


def gauss_kernel(X, Y=None, sigma=1.0):
    """
    Returns the Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))
    """
    sq_dists = sq_distances(X, Y)
    K = np.exp(-sq_dists / (2 * sigma ** 2))
    return K


# IMPLEMENT
def linear_kernel(X, Y):
    return np.dot(X, Y.T)

def HSIC_stat(X, Y):
    """
    Computes the HSIC test statistic with Gaussian kernel and bandwidth the median of Euclidean distances (X,Y),
    using the function median_bandwidth
    :param X: Dataset X nxq
    :param Y: Dataset Y nxl
    :return: The HSIC statistic, see [Gretton et al. 2007]
    """

    med = median_bandwidth(np.vstack((X, Y)))
    K_XY = np.exp(- sq_distances(X,Y) / (2 * med ** 2))

    K_XX_ = K_XY[:len(X),:len(X)]
    K_YY_ = K_XY[len(X):,len(X):]

    H = np.eye(len(K_XX_)) - 1.0 / len(K_XX_)
    statistic = np.trace(K_XX_.dot(H).dot(K_YY_.dot(H))) / (len(K_XX_) ** 2)
    return statistic, K_XY


def cov_center(X,Y, norm = 2.):
    """
    Returns the centered covariance statistic using p-norm, by default norm=2.
    """
    akl = cdist(X, Y,  'minkowski', p=norm)
    ak = np.asarray([np.sum(akl[i])/ len(X)  for i in range(len(akl))] ) # sum per line
    al = np.asarray( [np.sum(akl.T[i])/ len(X)  for i in range(len(akl.T))])  # sum per column
    cov = akl - ak - al + np.sum(ak) / (len(X) ** 2)

    return cov

def dcor(X,Y, norm = 2):

    assert X.ndim == Y.ndim
    VX = cov_center(X, X)
    VY = cov_center(Y, Y)
    VXY = np.sum(np.multiply(VX, VY))
    statistic = VXY / (norm(VX)*norm(VY))
    return statistic

def permutation_test_pval(statistic_fct, X, Y, norm = 2., num_permutations = 1000):
    """
    Performs the permutation procedure to estimate the null (1-alpha) quantile
    :param statistic_fct: function that computes the statistic
    :param X: Dataset X nxq
    :param Y: Dataset Y nxl
    :param norm: choice of p-norm
    :param num_permutations: number of random permutations sampled to approximate the null quantile
    :return: p-value and list of computed test statistics
    """
    assert X.ndim == Y.ndim

    statistics = np.zeros(num_permutations)
    my_test_raw = statistic_fct(X, Y)

    pval = 1.
    for i in range(num_permutations-1):
        perm_idx = np.random.permutation(len(Y))
        XY = np.vstack((X, Y[perm_idx]))
        my_test_perm = statistic_fct(X, Y[perm_idx], norm = norm)
        statistics[i] = my_test_perm
        if my_test_perm >= my_test_raw:
            pval += 1.

    pval = (1. /num_permutations) * pval

    return pval, statistics




def get_RTB_pval(data, n, m, u0, alpha, asymptotic=True):
    """
    :param data: univariate sample of size n+m
    :param n: length of the first sample
    :param m: length of the second sample
    :param u0: proportion of best ranked instances for RTB
    :param alpha: significance level of the test
    :param asymptotic: threshold based on the asymptotic distribution
    :return: p-value of the test statistic W_phi with RTB score-generating function of proportion u_0
    """
    N = n + m
    p = n / N
    phi = lambda x, u0: x / (N + 1.) if (x / (N + 1.) >= u0) else 0.0

    if asymptotic == True:
        E, V = (1 - u0 ** 2) / 2, p * (1 + (u0 ** 2) * (6 - 4 * u0 - 3 * (u0 ** 2))) / 12

    else:
        E = np.sum([phi(x, u0) / N for x in np.arange(1, N + 1., 1)])
        V = (1 / (N - 1)) * np.sum([phi(x, u0) ** 2 for x in np.arange(1, N + 1., 1)]) - (E ** 2)
        V = p * V

    W1 = (1 / n) * np.sum([phi(x, u0) for x in rankdata(data, method='average')][:n], axis=0)
    W = W1
    z = (W - E) / np.sqrt(V / n)
    _, pval = normtest(z, alternative='greater')

    thresh = distributions.norm.isf(alpha, loc=E, scale=V)
    return W, pval, thresh, E, V







