"""  author: Myrto Limnios, myli@math.ku.dk """

""" 
Main script to generate the synthetic data samples used in the paper Section Numerical Experiments:
  "On Ranking-based Tests of Independence"

Models : (GL), (GL+), (M1), (M1d), (M1s)
"""

import numpy as np
from numpy.random import multivariate_normal

def XY_generator_indep(n, m, d, rho, sample_type, rng):

    if sample_type == 'GL':
        Gamma = np.eye(d)
        for k in range(int(d/2)):
            Gamma[k, int(d/2):] = rho
            Gamma[int(d/2):, k] = rho

        Gamma = (1./np.sqrt(int(d)))*Gamma
        XYprod = multivariate_normal(np.zeros(d), Gamma, n)
        Xprod = np.asarray(XYprod[:,:int(d/2)].astype(np.float32))
        Yprod = np.asarray(XYprod[:,int(d/2):].astype(np.float32))

        print(Xprod.ndim, Yprod.ndim)

    if sample_type == 'GL+':
        Gamma = np.eye(d)
        Gamma[0:int(d/20), int(d/2):int(d*2/3)] = rho
        Gamma[int(d/2):int(d*2/3), 0:int(d/20)] = rho

        Gamma = (1. / np.sqrt(int(d))) * Gamma
        XYprod = rng.multivariate_normal(np.zeros(d), Gamma, n)
        Xprod = np.asarray(XYprod[:,:int(d/2)].astype(np.float32))
        Yprod = np.asarray(XYprod[:,int(d/2):].astype(np.float32))

        print(Xprod.ndim, Yprod.ndim)

    if sample_type == 'M1':

        if rho == 0.:
            X = rng.normal(0, 1, n+m)
            Y = rng.normal(0, 1, n+m)
            Xprod = np.hstack((X[:n, np.newaxis], rng.multivariate_normal(np.zeros(int(d / 2) - 1),  (1./np.sqrt(int(d/2)))*np.eye(int(d / 2) - 1), n)))
            Yprod = np.hstack((Y[:n, np.newaxis], rng.multivariate_normal(np.zeros(int(d / 2) - 1), (1./np.sqrt(int(d/2)))*np.eye(int(d / 2) - 1), n)))

        else:
            theta = rng.uniform(low=0., high=2 * np.pi, size=n)
            A = rho*np.ones(n)

            X = A * np.cos(theta) +  rng.normal(0, 1, n) / 4
            Y = A * np.sin(theta) +  rng.normal(0, 1, n) / 4
            Xprod = np.hstack((X[:,np.newaxis] , rng.multivariate_normal(np.zeros(int(d / 2) - 1), (1./np.sqrt(int(d/2)))*np.eye(int(d / 2) - 1), n) ))
            Yprod = np.hstack((Y[:,np.newaxis] , rng.multivariate_normal(np.zeros(int(d / 2) - 1), (1./np.sqrt(int(d/2)))*np.eye(int(d / 2) - 1), n) ))

    if sample_type == 'M1d':

        if rho == 0.:
            X = rng.normal(0, 1, n+m)
            Y = rng.normal(0, 1, n+m)
            Xprod = np.hstack((X[:n, np.newaxis], rng.uniform(low=0., high=1., size=(n, int(d / 2) - 1))))
            Yprod = np.hstack((Y[:n, np.newaxis], rng.uniform(low=0., high=1., size=(n, int(d / 2) - 1))))

        else:
            theta = rng.uniform(low=0., high=2 * np.pi, size=n)
            A = rho*np.ones(n)
            X = np.array(A * np.cos(theta) + rng.normal(0, 1, n) / 4)[:,np.newaxis]
            Y = np.array(A * np.cos(theta) + rng.normal(0, 1, n) / 4)[:,np.newaxis]

            for k in range(int(d/2-1)):
                    theta = rng.uniform(low=0., high=2 * np.pi, size=n)
                    A = rho*np.ones(n) #np.random.randint(int(rho), size=n)
                    x = np.array(A * np.cos(theta) + rng.normal(0, 1, n) / 4)[:,np.newaxis]
                    X= np.hstack((X,x))
                    y = np.array(A * np.cos(theta) + rng.normal(0, 1, n) / 4)[:,np.newaxis]
                    Y= np.hstack((Y,y))

            Xprod = X
            Yprod = Y

    if sample_type == 'M1s':

        if rho == 0.:
            X = rng.normal(0, 1, n + m)
            Y = rng.normal(0, 1, n + m)
            Xprod = np.hstack((X[:n, np.newaxis], rng.uniform(low=0., high=1., size=(n, int(d / 2) - 1))))
            Yprod = np.hstack((Y[:n, np.newaxis], rng.uniform(low=0., high=1., size=(n, int(d / 2) - 1))))

        else:
            theta = rng.uniform(low=0., high=2 * np.pi, size=n)
            A = rho * np.ones(n)
            X = np.array(A * np.cos(theta) + rng.normal(0, 1, n) / 4)[:, np.newaxis]
            Y = np.array(A * np.cos(theta) + rng.normal(0, 1, n) / 4)[:, np.newaxis]

            for k in range(int(d / 4 - 1)):
                theta = rng.uniform(low=0., high=2 * np.pi, size=n)
                A = rho * np.ones(n)
                x = np.array(A * np.cos(theta) + rng.normal(0, 1, n) / 4)[:, np.newaxis]
                X = np.hstack((X, x))
                y = np.array(A * np.cos(theta) + rng.normal(0, 1, n) / 4)[:, np.newaxis]
                Y = np.hstack((Y, y))

            Xprod = np.hstack((X, rng.uniform(low=0., high=1., size=(n, int(d /4)))))
            Yprod = np.hstack((Y, rng.uniform(low=0., high=1., size=(n, int(d /4)))))

    return Xprod, Yprod
