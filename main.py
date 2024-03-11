"""
author: Myrto Limnios, myli@math.ku.dk
"""

"""
Main script to run for the Independence testing problem for the article, Section Numerical Experiments (Sec. 4):
   "On Ranking-based Tests of Independence"
   
Use of the functions coded in stattest_fct
Python code for the probabilistic models in datagenerator for generating the two samples
The variables are denoted as in the main paper.

What it does:
 1. Samples two data samples from different distribution functions using datagenerator
 models available: (GL), (GL+), (M1), (M1d), (M1s)
 2. Performs Rank Forest bipartite ranking algorithms in the first halves to learning the optimal model (Steep 2)
 3. Uses the outputs of 2. to score the second halves to the real line (Step 3)
 4. Performs the hypothesis test on the obtained univariate two samples (Step 3)
 5. Compares the results to SoA algorithms: HSIC [Gretton et al. 2007],
               dCor with L1 and L2 norms [Szekely et al. 2004]
 6. Outputs the numerical p-values for each sampling loop
"""


import stattest_fct
import datagenerator as data
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import utils as utilsRT
import stattest_fct

import treerank as TR
from scipy.stats import  mannwhitneyu

import datetime

seed = 
rng = np.random.default_rng(seed)


'''  generate two-sample data '''
rho =  #dependence parameter
sample_type = ''  #GL, GL+, M1, M1s, M1d

depth =  #depth tree
K_tree =  #number of subsampled trees

K_perm =  #permutation samples for rank
B_perm =  #permutation pvalues benchmark

'''  Test parameters  '''
alpha =  #threshold of the tests

'''  Power estimation parameters MC samplings '''
B_pow =

'''  Implemented tests '''
name_methods = ['rForest_MWW', 'rForest_RTB95', 'rForest_RTB90', 'rForest_RTB85', 'HSIC', 'dCor_L2', 'dCor_L1']


rho_range = [] #range of the dependence parameter
N_range = [] #size of the TOTAL sample size
d_range = [] #dimension of the feature space q+l
param_range = rho_range


''' Type the path to your directory '''
path = r'/Users/.../'

for N in N_range:
    ntst, mtst = int(N/5), int(N/5)
    n, m = int(4*N/5), int(4*N/5)
    subspl_len = [ntst, mtst]

    print('sizes', n, ntst)

    for d in d_range:
        print('loop', 'sample size', 'd', d)

        str_param = sample_type + str(n) + str(m) + str(d) + str(K_tree) + str(K_perm) +  str(B_pow) + str(int(ntst)) + str(int(mtst)) +str(int(depth))

        for rho in np.around(rho_range, decimals=2):
            print('sizes', n, ntst, d, rho)

            '''  Generate the data: XY matrix of the twosamples, with scor=1 for XY product of marginals  and scor=0 for XY joint distribution '''
            dict_pval_all = dict(zip(name_methods, [[] for i in range(len(name_methods))]))

            Xprod_tr_, Yprod_tr_ = data.XY_generator_indep(n * B_pow, m * B_pow, d, rho, sample_type, rng)
            Xprod_tst_, Yprod_tst_ = data.XY_generator_indep(ntst * B_pow, mtst * B_pow, d, rho, sample_type, rng)

            s_train, scor_test = np.concatenate((np.ones(int(n)), -np.ones(int(m)))).astype(np.float32), np.concatenate(
                (np.ones(int(ntst)), -np.ones(int(mtst)))).astype(np.float32)

        for b in range(B_pow):
                """ subsampling multivariate two samples for the Rank tests """
                Xprod_tr, Yprod_tr = np.asarray(Xprod_tr_[n*b:int(n*(b+1))]), np.asarray(Yprod_tr_[n*b:int(n*(b+1))])
                Xprod_tst, Yprod_tst = np.asarray(Xprod_tst_[ntst * b:int(ntst * (b + 1))]), np.asarray(Yprod_tst_[ntst * b:int(ntst * (b + 1))])

                XYjoint_tr, XYjoint_tst = np.hstack((Xprod_tr, Yprod_tr)), np.hstack((Xprod_tst, Yprod_tst))

                assert len(Xprod_tr) == len(Yprod_tr)
                for kperm in range(K_perm):
                    """" Construction of the training sample for Step 2"""
                    perm_idx_trainY = np.random.permutation(n)

                    assert len(perm_idx_trainY) == len(Yprod_tr)

                    XYprod_tr = np.hstack((Xprod_tr, Yprod_tr[perm_idx_trainY]))
                    XY_tr = np.vstack((XYjoint_tr, XYprod_tr))

                    assert len(XYjoint_tr) == len(XYprod_tr)

                    """" Construction of the testing sample for Step 3"""
                    perm_idx_tstY = np.random.permutation(int(mtst))
                    XYprod_tst = np.hstack((Xprod_tst, Yprod_tr[perm_idx_tstY]))
                    XY_tst = np.vstack((XYjoint_tst, XYprod_tst))

                    print('len test', len(XY_tst))
                    assert len(XYjoint_tst) == len(XYprod_tst)

                    y_pred_tr = np.zeros(int((ntst + mtst)))
                    s_predrk_list = []

                    print("#" * 80, "#{:^78}#".format("TREE"), "#" * 80, sep='\n')
                    for kt in range(K_tree):
                        assert len(XY_tr) == len(s_train)

                        XY_index = np.arange(0, len(XY_tr))
                        XY_index_train, _, scor_train_t, _ = train_test_split(XY_index, s_train, test_size=0.05,
                                                                              stratify=s_train)
                        xtrain_temp = XY_tr[XY_index_train]
                        tree = TR.TreeRANK(max_depth=depth, verbose=0)
                        tree.fit(xtrain_temp, scor_train_t)

                        """" Predict the scores of the test sample for Step 3 """
                        y_pred_tr += tree.predict_scor(XY_tst) / K_tree

                    ind = 0
                    s_predrk_list = [y_pred_tr]
                    for ypred in s_predrk_list:
                        sx = ypred[np.where(scor_test == 1)].tolist()
                        sy = ypred[np.where(scor_test == -1)].tolist()

                        if (sx == sy) == True:
                            print('same prediction')
                            dict_pval_all[name_methods[ind]][b] += 1.0/ K_perm
                            dict_pval_all[name_methods[ind + 1]][b] += 1.0/ K_perm
                            dict_pval_all[name_methods[ind + 2]][b] += 1.0/ K_perm
                            dict_pval_all[name_methods[ind + 3]][b] += 1.0/ K_perm
                        else:
                            mww, mww_pR = mannwhitneyu(sx, sy, use_continuity=True, alternative='greater')
                            W95, pval95, _, _, _ = utilsRT.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.95, alpha, asymptotic=True)
                            W9, pval9, _, _, _ = utilsRT.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.90, alpha, asymptotic=True)
                            W85, pval85, _, _, _ = utilsRT.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.85, alpha, asymptotic=True)
                            dict_pval_all[name_methods[ind]][b] += mww_pR / K_perm
                            dict_pval_all[name_methods[ind + 1]][b] += pval95 / K_perm
                            dict_pval_all[name_methods[ind + 2]][b] += pval9 / K_perm
                            dict_pval_all[name_methods[ind + 3]][b] += pval85 / K_perm

                        ind +=1

                X_oth = np.vstack((Xprod_tr, Xprod_tst))
                Y_oth = np.vstack((Yprod_tr, Yprod_tst))

                dict_pval_all['HSIC'].append(stattest_fct.permutation_test_pval(stattest_fct.HSIC_stat, X_oth, Y_oth,num_permutations=B_perm)[0])
                dict_pval_all['dCor_L2'].append(stattest_fct.permutation_test_pval(stattest_fct.dcor, X_oth, Y_oth,num_permutations=B_perm)[0])
                dict_pval_all['dCor_L1'].append(stattest_fct.permutation_test_pval(stattest_fct.dcor, X_oth, Y_oth, norm = 1,  num_permutations=B_perm)[0])

            ''' Save file to local directory '''
            df_pvalue_MWW = pd.DataFrame.from_dict(dict_pval_all)
            df_pvalue_MWW.to_csv(path + 'pval_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + 'rho' + str(rho) + '.csv')
            df_pvalue_MWW = {}

            print('loop', 'sample size', 'd', d)
