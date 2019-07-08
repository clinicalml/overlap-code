# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Fredrik D. Johansson
# ----------------------------------------------#
#

import sys, os
import numpy as np
sys.path.append('..')

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, datasets
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_recall_curve, accuracy_score

from overrule.overrule import OverRule2Stage
from overrule.baselines import knn, marginal, propscore, svm
from overrule.support import SVMSupportEstimator, SupportEstimator
from overrule.overlap import SupportOverlapEstimator
from overrule.ruleset import BCSRulesetEstimator, RulesetEstimator

if __name__ == '__main__':
    # Construct output folder
    os.makedirs('../results/iris', exist_ok=True)

    # Load dataset
    iris = datasets.load_iris()

    # Define column names
    f_cols = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    
    # Construct dataframe
    D = pd.DataFrame(iris['data'], columns=f_cols)
    
    # Define label
    y = iris['target']
    D['class'] = y
    
    # Select only class 1 and 2
    Dg = D.loc[D['class']>0,:].copy()
    
    # Define group indicator
    Dg['group'] = Dg['class']-1
    
    # Define base estimator
    exp_label = 'knn'
    O = knn.KNNOverlapEstimator(k=8)

    # Define rule estimators
    RS_s = BCSRulesetEstimator(n_ref_multiplier=1, alpha=.9, lambda0=.7, lambda1=0)
    RS_o = BCSRulesetEstimator(n_ref_multiplier=0, alpha=.9, lambda0=.7, lambda1=0)

    # Fit overlap estimator
    M = OverRule2Stage(O, RS_o, RS_s)
    M.fit(Dg[f_cols], Dg['group'])
    
    # Get learned rules
    rules = M.rules(as_str=True)
    open('../results/iris/rules.txt', 'w').write('Support:\n'+rules[0]+'\nPropensity:\n'+rules[1])
    
    # Plot rules
    plt.rc('font', size=18, family='serif')
    R = M.rules()[1]
    O = M.predict(Dg[f_cols])
    
    l = 0
    fig, axs = plt.subplots(2,3,figsize=(12,8))
    for i in range(len(f_cols)):
        for j in range(i+1, len(f_cols)):

            f1 = f_cols[i]
            f2 = f_cols[j]

            axs[int(l/3), l%3].scatter(Dg[f1], Dg[f2], c=Dg['group'], cmap='bwr', zorder=2)
            axs[int(l/3), l%3].scatter(Dg.iloc[O==1][f1], Dg.iloc[O==1][f2], edgecolors='k', s=200, alpha=1., facecolors='none')

            axs[int(l/3), l%3].set_xlabel(f1)
            axs[int(l/3), l%3].set_ylabel(f2)

            for k in R:
                x0l = D[f1].min()-2
                x0u = D[f1].max()+2
                x1l = D[f2].min()-2
                x1u = D[f2].max()+2
                for a in k:
                    if a[0] == f1 and a[1] == '>':
                        x0l = a[2]
                    if a[0] == f1 and a[1] == '<=':
                        x0u = a[2]
                    if a[0] == f2 and a[1] == '>':
                        x1l = a[2]
                    if a[0] == f2 and a[1] == '<=':
                        x1u = a[2]

    #            color = (.3,0,.3)
                color = 'g'
                rect = patches.Rectangle((x0l,x1l),x0u-x0l,x1u-x1l,linewidth=1,
                                         edgecolor=color, facecolor=color, alpha=.4, zorder=-4)
                axs[int(l/3), l%3].add_patch(rect)
            l += 1
    plt.tight_layout()
    plt.savefig('../results/iris/iris_%s.pdf' % exp_label)
    plt.close()
