# independence_ranktest

Code used for the numerical experiments of the companion working paper, Section Numerical Experiments:
   "On Ranking-based Tests of Independence".
   
# Information

Use the main.py script to run for the Independence test.

The functions coded in stattest_fct and utils are needed for the execution.

Code the probabilistic models in datagenerator for generating the two samples.

The variables are denoted as in the main paper Section Numerical Experiments.

author:


# What it does:
 1. Samples two dependent data samples using datagenerator
 2. Performs a Random Forest bipartite ranking algorithm in the first halves to learning the optimal model

 3. Uses the outputs of 2. to score the second halves to the real line
 4. Performs the hypothesis test on the obtained univariate two samples
 5. Compares the results to SoA algorithms: Hilbert-Schmidt Indeependence Criterion [Gretton et al. 2007](https://papers.nips.cc/paper_files/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html) with Gaussian kernel of bandwidth the median heuristic choice,
               Energy statistic [Szekely et al. 2007] coded in stattest_fct
 6. Outputs the numerical pvalue for each sampling loop


# Requirements

Python librairies: numpy, pandas, scipy, sklearn, random 
