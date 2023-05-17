
import numpy as np
from scipy.stats import rankdata
from scipy.stats import distributions


def get_RTB_pval(data, n, m, u0, alpha, asymptotic=True):
    N = n + m
    p = n / N
    phi = lambda x, u0: x / (N + 1.) if (x / (N + 1.) >= u0) else 0.0

    if asymptotic == True:
        E, V = (1 - u0 ** 2) / 2, p * (1 + (u0 ** 2) * (6 - 4 * u0 - 3 * (u0 ** 2))) / 12

    else:
        E = np.sum([phi(x, u0) / N for x in np.arange(1, N + 1., 1)])  ##   E(1/n)*What
        V = (1 / (N - 1)) * np.sum([phi(x, u0) ** 2 for x in np.arange(1, N + 1., 1)]) - (E ** 2)  ## Var(1/n)*What
        V = p * V

    W1 = (1 / n) * np.sum([phi(x, u0) for x in rankdata(data, method='average')][:n], axis=0)
    #W2 = (1 / m) * np.sum([phi(x, u0) for x in rankdata(data, method='average')][n:], axis=0)
    #W = max(W1,W2)
    W = W1
    z = (W - E) / np.sqrt(V / n)
    _, pval = _normtest_finish(z, alternative='greater')

    thresh = distributions.norm.isf(alpha, loc=E, scale=V)

    return W, pval, thresh, E, V


def _normtest_finish(z, alternative):
    """Common code between all the normality-test functions.
    author: scipy"""
    if alternative == 'less':
        prob = distributions.norm.cdf(z)
    elif alternative == 'greater':
        prob = distributions.norm.sf(z)
    elif alternative == 'two-sided':
        prob = 2 * distributions.norm.sf(np.abs(z))
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")

    if z.ndim == 0:
        z = z[()]

    return z, prob