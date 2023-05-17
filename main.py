# Main script to run for the Independence testing problem
# Use of the functions coded in stattest_fct
# Code the probabilistic models in datagenerator for generating the two samples
# The variables are denoted as in the main paper Section Numerical Experiments:
#    "On Ranking-based Tests of Independence"

# author:

# What it does:
# 1. Samples two data samples from different distribution functions using datagenerator
# 2. Performs Rank Forest bipartite ranking algorithms in the first halves to learning the optimal model

# 3. Uses the outputs of 2. to score the second halves to the real line
# 4. Performs the hypothesis test on the obtained univariate two samples
# 5. Compares the results to SoA algorithms: HSIC [Gretton et al. 2007],
#               Energy statistic [Szekely et al. 2004]
# 6. Outputs the numerical pvalue for each sampling loop



import stattest_fct
import datagenerator as data
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import utils as utilsRT

from scipy.stats import  mannwhitneyu

import datetime

seed = # choose your seed
rng = np.random.default_rng(seed)


'''  generate two-sample data '''
eps =  #dependence parameter
sample_type = ''  #blobs_loc, blobs_rot, simple

d_ = #dimension of the feature space q+l
depth =  #depth tree
K_tree =  #number of subsampled trees

K_perm =  #permutation samples for rank
B_perm =  #permutation pvalues benchmark

'''  Test parameters  '''
alpha =  #threshold of the tests

'''  Power estimation parameters MC samplings '''
B_pow =

'''  Benchmark tests parameters '''
list_names = ['HSIC', 'dCor']

eps_range = []
param_range = eps_range


'''   Complete with the path to your directory '''
path_MWW = r'/Users/.../'

k = 0

name_BR = []

subspl_len = []
ntst, mtst = 0, 0
n, m = 0, 0
sim, str_param = 0, 0


for N in [2000]:

    ntst, mtst = int(N/10), int(N/10)
    n, m = int(4*N/10), int(4*N/10)
    subspl_len = [ntst, mtst]

    print('sizes', n, ntst)

    for d in [d_]:

        print('loop', 'sample size', 'd', d)

        sim = sample_type + str(n) + str(m) + str(d) + str(K_tree) + str(K_perm) +  str(B_pow) + str(int(ntst)) + str(int(mtst)) +str(int(depth))
        str_param = sim

        for eps in np.around(eps_range, decimals=2):
            print('sizes', n, ntst, d, eps)

            '''  Generate the data: XY matrix of the twosamples, with scor=1 for X and scor=0 for Y, q=unit vector '''
            print('Generate datasets')

            pwr_rk = np.zeros((1, len(name_BR)))
            dict_pvalue_MWW = dict(zip(name_BR, [np.zeros(B_pow) for i in range(len(name_BR))]))

            pwr_othr_ = np.zeros((1, len(list_names) - 1))
            dict_pval_othr = dict(zip(list_names, [[] for i in range(len(list_names))]))

            Xprod_tr_, Yprod_tr_, XYjoint_tr_ = data.XY_generator_indep(n*B_pow, m*B_pow, d, eps, sample_type, rng)

            print('len samples', len(Xprod_tr_),len(Yprod_tr_), len(XYjoint_tr_))
            Xprod_tst_, Yprod_tst_, XYjoint_tst_ = data.XY_generator_indep(ntst*B_pow, mtst*B_pow, d, eps, sample_type, rng)
            s_train, scor_test = np.concatenate((np.ones(int(m)), -np.ones(int(n)))).astype(np.float32), np.concatenate((np.ones(int(mtst)),-np.ones(int(ntst)))).astype(np.float32)

            for b in range(B_pow):
                """ subsampling multivariate two samples for the Rank tests """

                Xprod_tr, Yprod_tr = np.asarray(Xprod_tr_[n*b:int(n*(b+1))]), np.asarray(Yprod_tr_[n*b:int(n*(b+1))])
                Xprod_tst, Yprod_tst = np.asarray(Xprod_tst_[ntst * b:int(ntst * (b + 1))]), np.asarray(Yprod_tst_[ntst * b:int(ntst * (b + 1))])

                XYjoint_tr, XYjoint_tst = np.asarray(XYjoint_tr_[m*b:int(m*(b+1))]), np.asarray(XYjoint_tst_[mtst*b:int(mtst*(b+1))])

                print(XYjoint_tr[0], XYjoint_tst[0], Xprod_tr[0], Yprod_tr[0], len(Yprod_tr))

                assert len(Xprod_tr) == len(Yprod_tr)


                for kperm in range(K_perm):
                    """" Construction of the training sample"""
                    perm_idx_trainY = np.random.permutation(n)

                    print(len(perm_idx_trainY), len(Yprod_tr))
                    assert len(perm_idx_trainY) == len(Yprod_tr)

                    XYprod_tr = np.hstack((Xprod_tr, Yprod_tr[perm_idx_trainY]))

                    print(XYprod_tr.ndim,XYjoint_tr.ndim)

                    XY_tr = np.vstack((XYjoint_tr, XYprod_tr))

                    print('len train', len(XY_tr))
                    assert len(XYjoint_tr) == len(XYprod_tr)

                    """" Construction of the testing sample"""
                    perm_idx_tstY = np.random.permutation(int(mtst))

                    XYprod_tst = np.hstack((Xprod_tst, Yprod_tr[perm_idx_tstY]))

                    #XY_tst = np.vstack((XYprod_tst, XYjoint_tst))
                    XY_tst = np.vstack((XYjoint_tst, XYprod_tst))

                    print('len test', len(XY_tst))
                    assert len(XYjoint_tst) == len(XYprod_tst)

                    print(XYprod_tr.ndim, XYprod_tst.ndim, len(XYprod_tr), len(XYprod_tst))
                    print('indices train from', n*b, 'to',  n*(b+1), 'from',  n*B_pow + m*b, 'to', n*B_pow + m*(b+1))
                    print('indices test', ntst*b, ntst*B_pow + mtst*b, ntst*B_pow + mtst*(b+1))

                    y_pred_tr = np.zeros(int((ntst + mtst)))
                    s_predrk_list = []

                    print("#" * 80, "#{:^78}#".format("TREE"), "#" * 80, sep='\n')
                    for kt in range(K_tree):
                        #print('perm round', kperm, 'tree round', kt)
                        assert len(XY_tr) == len(s_train)

                        XY_index = np.arange(0, len(XY_tr))
                        XY_index_train, _, scor_train_t, _ = train_test_split(XY_index, s_train, test_size=0.05,
                                                                              stratify=s_train)
                        xtrain_temp = XY_tr[XY_index_train]

                        tree = TR.TreeRANK(max_depth=depth, verbose=0, C=100.0, penalty='l2', fit_intercept=True)
                        tree.fit(xtrain_temp, scor_train_t)

                        """" Predict """
                        y_pred_tr += tree.predict_scor(XY_tst) / K_tree

                    ind = 0
                    s_predrk_list = [y_pred_tr]
                    for ypred in s_predrk_list:
                        sx = ypred[np.where(scor_test == 1)].tolist()
                        sy = ypred[np.where(scor_test == -1)].tolist()

                        if (sx == sy) == True:
                            print('same prediction')
                            dict_pvalue_MWW[name_BR[ind]][b] += 1.0/ K_perm
                            dict_pvalue_MWW[name_BR[ind + 1]][b] += 1.0/ K_perm
                            dict_pvalue_MWW[name_BR[ind + 2]][b] += 1.0/ K_perm
                            dict_pvalue_MWW[name_BR[ind + 3]][b] += 1.0/ K_perm
                        else:
                            mww, mww_pR = mannwhitneyu(sx, sy, use_continuity=True, alternative='greater')
                            print(mannwhitneyu(sx, sy, use_continuity=True, alternative='greater'))
                            W9, pval9, _, _, _ = utilsRT.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.9, alpha, asymptotic=True)
                            W8, pval8, _, _, _ = utilsRT.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.95, alpha, asymptotic=True)
                            W7, pval7, _, _, _ = utilsRT.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.99, alpha, asymptotic=True)
                            dict_pvalue_MWW[name_BR[ind]][b] += mww_pR / K_perm
                            # dict_pvalue_MWWR[name_BR[ind]].append(mww_pR)
                            dict_pvalue_MWW[name_BR[ind + 1]][b] += pval9 / K_perm
                            dict_pvalue_MWW[name_BR[ind + 2]][b] += pval8 / K_perm
                            dict_pvalue_MWW[name_BR[ind + 3]][b] += pval7 / K_perm
                        ind +=1
                print(dict_pvalue_MWW)
                scor_oth = np.concatenate((s_train, scor_test))

                X_oth = np.vstack((XYjoint_tr[:,:int(d/2)], Xprod_tr, XYjoint_tst[:,:int(d/2)], Xprod_tst))
                Y_oth = np.vstack(( XYjoint_tr[:,int(d/2):], Yprod_tr,XYjoint_tst[:,int(d/2):], Yprod_tst))
                print(len(X_oth), X_oth.ndim)
                dict_pval_othr['HSIC'].append(stattest_fct.permutation_test_pval(stattest_fct.HSIC_stat, X_oth, Y_oth,num_permutations=B_perm)[0])
                dict_pval_othr['dCor'].append(stattest_fct.permutation_test_pval(stattest_fct.dcor, X_oth, Y_oth,num_permutations=B_perm)[0])


            ''' Power estimation for all methods '''

            df_pvalue_MWW = pd.DataFrame.from_dict(dict_pvalue_MWW)
            df_pvalue_MWW.to_csv(path_MWW + 'pval_MWW_all_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + 'eps' + str(eps) + '.csv')
            df_pvalue_MWW = {}

            df_pvalue_SOA = pd.DataFrame.from_dict(dict_pval_othr)
            df_pvalue_SOA.to_csv(path_MWW + 'pval_SoA_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + 'eps' + str(eps) + '.csv')
            df_pvalue_SOA = {}

            print('loop', 'sample size', 'd', d)
