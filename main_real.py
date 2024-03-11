"""
author: Myrto Limnios, myli@math.ku.dk
"""


"""
Main script to run the independence testing problem applied to real data analysis for statistical parity (Sec. 4) from 
https://github.com/feedzai/bank-account-fraud

Use of the functions coded in stattest_fct

The variables are denoted as in the main paper Section Numerical Experiments:
   "On Ranking-based Tests of Independence"

What it does:
 1. Import the datasamples for independence testing problem applied to real data analysis for statistical parity (Sec. 4)
 2. Performs Rank Forest bipartite ranking algorithms in the first halves to learning the optimal model
 3. Uses the outputs of 2. to score the second halves to the real line
 4. Performs the hypothesis test on the obtained univariate two samples
 6. Outputs the numerical pvalue for each fold
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

import treerank as TR
import utils_ranktest as utilsRT
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score

from scipy.stats import mannwhitneyu
import datetime

seed = 24
rng = np.random.RandomState(seed)

""" rForest parameters """
depth =
K_tree =

B_pow =   # MC samplings
K_BR =  # permutation samples for rForest

'''  Test parameters  '''
alpha = 0.05  # threshold of the tests

'''  Name of the methods used for the associated dictionary  '''
name_methods = ['rForest_MWW', 'rForest_RTB95', 'rForest_RTB90', 'rForest_RTB85']


""" DATA IMPORT AND PREPROCESSING"""
data = pd.DataFrame(pd.read_csv(r'/Users/.../archive_neurips22/Base_subsample.csv')).replace('?', np.nan).dropna()
data, _ = train_test_split(data, test_size=0.99, stratify=data[['fraud_bool']],  random_state=rng)

print('number fraud', np.sum(data[['fraud_bool']]), len(data))

""" Set up the parameters for the cross-validation """
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits)

data = pd.get_dummies(data, drop_first=True).drop(['zip_count_4w','bank_branch_count_8w'], axis=1)


dict_pval_all = dict(zip(name_methods, [[] for i in range(len(name_methods))]))
ypred = []

""" Learning the rule and testing """
for b, (train, test) in enumerate(cv.split(data, data['fraud_bool'])):
    data_train, data_test = data.iloc[train], data.iloc[test]

    data_train_cov = data_train.drop(['fraud_bool'], axis=1)
    data_test_cov = data_test.drop(['fraud_bool'], axis=1)

    mod = RandomForestClassifier()
    yp_RF = mod.fit(data_train_cov, data_train['fraud_bool']).predict_proba(data_test_cov)
    yc_RF = mod.fit(data_train_cov, data_train['fraud_bool']).predict(data_test_cov)
    print("r2_score:", r2_score(data_test['fraud_bool'], yc_RF), 'mse', ((yc_RF - data_test['fraud_bool']) ** 2).mean())

    ypred.append(yp_RF)
    data_test_sens = data_test_cov[['customer_age', 'date_of_birth_distinct_emails_4w' , 'name_email_similarity']]

    assert len(yp_RF) == len(data_test_sens)

    ind = 0
    for yp in ypred:

        data_X = np.array([a[0] for a in np.array(yp)])[:,np.newaxis]
        data_Y = np.array(data_test_sens)

        ranking_idx, testing_idx = train_test_split(range(len(data_X)), test_size=int(len(data_X)/5), stratify=data_test['fraud_bool'])
        print('len rank/test', len(ranking_idx), len(testing_idx))

        data_train_X, data_test_X, data_train_Y, data_test_Y = data_X[ranking_idx], data_X[testing_idx], data_Y[ranking_idx], data_Y[testing_idx]
        print('len rank/test', len(data_train_X), len(data_test_X), len(data_train_Y), len(data_test_Y))

        for k in range(len(name_methods)):
            dict_pval_all[name_methods[k]].append(0.)

        for kperm in range(K_BR):

            """ Shuffling train sample for Step 2 """
            data_train_idx1, data_train_idx0 = train_test_split(range(len(data_train_Y)), test_size=0.5)
            perm_idx_trainY = np.random.permutation(data_train_idx1)

            data_train_XYjoint = np.hstack((data_train_X[data_train_idx0], data_train_Y[data_train_idx0]))
            data_train_XYperm = np.hstack((data_train_X[data_train_idx1], data_train_Y[perm_idx_trainY]))

            XY_tr = np.vstack((data_train_XYjoint, data_train_XYperm))

            """ Shuffling test sample for Step 3 """
            data_test_idx1, data_test_idx0 = train_test_split(range(len(data_test_Y)), test_size=0.5)
            perm_idx_testY = np.random.permutation(data_test_idx1)

            data_test_XYjoint = np.hstack((data_test_X[data_test_idx0], data_test_Y[data_test_idx0]))
            data_test_XYperm = np.hstack((data_test_X[data_test_idx1], data_test_Y[perm_idx_testY]))

            XY_tst = np.vstack((data_test_XYjoint, data_test_XYperm))

            """ Setting the dimensions and labels for Step 2 """
            n, m = len(data_train_XYjoint), len(data_train_XYperm)
            d_X, d_Y = len(data_train_X.T), len(data_train_Y.T)
            ntst, mtst = len(data_test_XYjoint), len(data_test_XYperm)

            scor_train = np.concatenate((np.ones(int(n)), -np.ones(int(m)))).astype(np.float32)
            scor_test = np.concatenate((np.ones(int(ntst)), -np.ones(int(mtst)))).astype(np.float32)

            y_pred_tr = np.zeros(int((ntst + mtst)))

            """ Train rForest based on K_tree replications """
            for kt in range(K_tree):
                assert len(XY_tr) == len(scor_train)

                XY_index = np.arange(0, len(XY_tr))
                XY_index_train, _, scor_train_t, _ = train_test_split(XY_index, scor_train, test_size=0.05,
                                                                      stratify=scor_train)
                xtrain_temp = XY_tr[XY_index_train]
                tree = TR.TreeRANK(max_depth=depth, verbose=0)
                tree.fit(xtrain_temp, scor_train_t)

                """" Predict the labels on the test sample for Step 3"""
                y_pred_tr += tree.predict_scor(XY_tst) / K_tree


            sx = y_pred_tr[np.where(scor_test == 1)].tolist()
            sy = y_pred_tr[np.where(scor_test == -1)].tolist()

            """ Compute the pvalues for the predictions, using MWW, RTB(u_0) """
            if (sx == sy) == True:
                dict_pval_all[name_methods[ind]][b] += 1.0 / K_BR
                dict_pval_all[name_methods[ind + 1]][b] += 1.0 / K_BR
                dict_pval_all[name_methods[ind + 2]][b] += 1.0 / K_BR
                dict_pval_all[name_methods[ind + 3]][b] += 1.0 / K_BR
            else:
                mww, mww_pR = mannwhitneyu(sx, sy, use_continuity=True, alternative='greater')
                W95, pval95, _, _, _ = utilsRT.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.95, alpha,
                                                          asymptotic=True)
                W90, pval90, _, _, _ = utilsRT.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.9, alpha,
                                                          asymptotic=True)
                W85, pval85, _, _, _ = utilsRT.get_RTB_pval(np.concatenate((sx, sy)), ntst, mtst, 0.85, alpha,
                                                          asymptotic=True)
                dict_pval_all[name_methods[ind]][b] += mww_pR / K_BR
                dict_pval_all[name_methods[ind + 1]][b] += pval95 / K_BR
                dict_pval_all[name_methods[ind + 2]][b] += pval90 / K_BR
                dict_pval_all[name_methods[ind + 3]][b] += pval85 / K_BR

        ind += 1
    print(dict_pval_all)



''' Saving the results '''
print('all', dict_pval_all)
df_pval_all = pd.DataFrame.from_dict(dict_pval_all)
df_pval_all.to_csv(r'/Users/.../pval_fraud_' '_'+ 'rep' + \
                   str(n_splits) + str(K_BR) + '_'  + '_' + datetime.datetime.today().strftime("%m%d%H%M") + '.csv')
df_pval_all = {}
