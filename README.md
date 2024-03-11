# independence_ranktest

Code used for the numerical experiments of the companion working paper, Section Numerical Experiments (Sec.4):
   "On Ranking-based Tests of Independence". 
   
Authors: Myrto Limnios, Stephan Clémençon.
# Information

1. Use the main.py script to run for the Independence test based on synthetic datasets, of probabilistic models implemented in datagenerator.py.
2. Use the main_real.py script to run for the Independence test based on real datasets using k-fold cross validation.

The functions coded in stattest_fct are needed for the execution of the proposed ranking-based method and state-of-the-art methods.

The variables are denoted as in the main paper Section Numerical Experiments.

# What it does:
 1. Samples two dependent data samples using datagenerator
 2. Performs a Random Forest bipartite ranking algorithm in the first halves to learning the optimal model (Step 2 of the proposed method)
 3. Uses the outputs of 2. to score the second halves to the real line (Step 3 of the proposed method)
 4. Performs the hypothesis test on the obtained univariate two samples (Step 3 of the proposed method)
 5. Compares the results to SoA algorithms: Hilbert-Schmidt Indeependence Criterion [Gretton et al. 2007](https://papers.nips.cc/paper_files/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html) with Gaussian kernel of bandwidth the median heuristic choice,
               Energy statistic with L1 and L2 norms [Szekely et al. 2007](https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-6/Measuring-and-testing-dependence-by-correlation-of-distances/10.1214/009053607000000505.full) coded in stattest_fct
 6. Outputs the numerical pvalue for each sampling loop

# Requirements

Python librairies: numpy, pandas, scipy, sklearn, random 
