#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
import synth_utils as utils
import shelve

from data import advCube
import sys; sys.path.insert(0, '../..')
from overrule.ruleset import BCSRulesetEstimator, RulesetEstimator

from sacred import Experiment
ex = Experiment('synthetic_removal')

import logging
from datetime import datetime

# Create a logger
logger = logging.getLogger('synthetic_removal')

# Add a logging file
logpath = './logs'
logtime = datetime.now().strftime('%Y-%m-%d-%H%M:%S:%f')

fh = logging.FileHandler('{}/{}.log'.format(logpath, logtime))
fh.setLevel(logging.INFO)

formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Tie this to the experiment
ex.logger = logger

@ex.config
def my_config():
    SEED=0
    TUNE_XI_SUPPORT=True
    COVERAGE=True
    CNF=True
    CNFSynth=True
    VERBOSE=True
    FIX_MARGINAL=False

    ALPHA=0.98
    N_REF_MULT=0.3
    LAMBDA0=1e-5
    LAMBDA1=1e-3

    D=20  # Maximum extra rules per beam seach iteration
    K=20  # Maximum results returned during beam search
    B=5  # Width of Beam Search

    nr = 2  # Rare Features
    rare_prob = 0.01  # Rare Feature Probability
    ni = 2   # Interaction Features
    nn = 6 # Normal (i.e., prob 0.5) Features
    nd = nn + ni + nr  # Total Features

@ex.capture
def get_support_estimator(cat_cols,
        VERBOSE, N_REF_MULT, ALPHA, LAMBDA0, LAMBDA1, CNF, D, B, K,
        _log):
    ex.info['n_ref_mult'] = N_REF_MULT
    ex.info['alpha'] = ALPHA
    ex.info['lambda0'] = LAMBDA0
    ex.info['lambda1'] = LAMBDA1
    ex.info['cnf'] = 1. if CNF else 0.
    ex.info['D'] = D
    ex.info['K'] = K
    ex.info['B'] = B

    return BCSRulesetEstimator(
            n_ref_multiplier=N_REF_MULT, cat_cols=cat_cols,
            alpha=ALPHA, gamma=1, eps=1e-4,
            lambda0=LAMBDA0, lambda1=LAMBDA1, D=D, B=B, K=K,
            solver='ECOS', iterMax=5,
            CNF=CNF, logger=_log, silent=(not VERBOSE))

@ex.capture
def get_data(rare_prob, nd, ni, nr, CNFSynth, FIX_MARGINAL, SEED, _log):
    _log.info("Generating fake data")
    ex.info['nr'] = nr
    ex.info['ni'] = ni
    ex.info['nd'] = nd
    ex.info['rare_prob'] = rare_prob
    ex.info['CNFSynth'] = CNFSynth
    ex.info['fix_marginal'] = FIX_MARGINAL
    return advCube(
            rare_prob=rare_prob,
            nd=nd, ni=ni, nr=nr,
            CNF=CNFSynth, seed=SEED, fix_marginal=FIX_MARGINAL)

@ex.automain
def run(SEED, TUNE_XI_SUPPORT,
        COVERAGE,
        CNF,
        VERBOSE, _log):

    _log.info("Starting Experiment")
    np.random.seed(SEED)
    w_eps = 1e-8

    X_train, check_incl = get_data()
    cat_cols = X_train.columns.values.tolist()
    # This is bit clunky, doesn't really get used, we always cast to ones
    Y_train = np.ones(X_train.shape[0])

    RS_s = get_support_estimator(cat_cols)\
            .fit(X_train, np.ones(Y_train.shape[0]))

    ex.info['lp_obj'] = RS_s.M.lp_obj_value
    ex.info['wlogged'] = np.log(RS_s.M.w_raw + w_eps)
    ex.info['w'] = RS_s.M.w_raw
    ex.info['z_index'] = RS_s.M.z.index.values.tolist()
    ex.info['z_values'] = RS_s.M.z.values

    if VERBOSE:
        _log.info("Reference / Actual Samples: {} / {}".format(
            RS_s.refSamples.shape[0], X_train.shape[0]))

    # Round the rules
    data_round = pd.concat([X_train, RS_s.refSamples], axis=0)
    o_round = np.hstack([np.ones(X_train.shape[0]),
                         -np.ones(RS_s.refSamples.shape[0])])

    ROUNDING = 'coverage' if COVERAGE else 'greedy'
    ex.info['coverage_rounding'] = COVERAGE

    RS_s.round_(data_round, o_round, ROUNDING, xi=0.1, use_lp=True)

    if VERBOSE:
        _log.info("Finding best xi for support")
    best_xi = 1

    if TUNE_XI_SUPPORT:
        if VERBOSE:
            _log.info("Finding best Xi for Support")
        best_xi = None
        best_auc = 0
        for xi in np.logspace(np.log10(0.1), .1, 5):
            if VERBOSE:
                _log.info("Trying xi: {:.2f}".format(xi))
            data_round = pd.concat([X_train, RS_s.refSamples], 0)
            o_round = np.hstack([np.ones_like(Y_train), -np.ones(RS_s.refSamples.shape[0])])
            RS_s.round_(data_round, o_round, ROUNDING, xi=xi, use_lp=True) 

            auc = roc_auc_score((o_round == 1).astype(int), RS_s.predict(data_round))

            if VERBOSE:
                _log.info('AUC (reference vs support): %.2f' % (auc))
            if auc > best_auc:
                best_xi = xi
                best_auc = auc
    else:
        if VERBOSE:
            _log.info("Using default Xi: {}".format(best_xi))

    if VERBOSE:
        _log.info("Rounding rules based on {} strategy".format(ROUNDING))

    data_round = pd.concat([X_train, RS_s.refSamples], 0)
    o_round = np.hstack([np.ones_like(Y_train), -np.ones(RS_s.refSamples.shape[0])])
    RS_s.round_(data_round, o_round, ROUNDING, xi=best_xi, use_lp=True)

    ex.info['rounded_obj'] = float(RS_s.get_objective_value_(data_round, o_round))

    # Basic statistics
    if VERBOSE:
        _log.info('Number of reference samples: {}'.format(RS_s.refSamples.shape[0]))
        _log.info('Coverage of data points: %.3f, Requested >= %.3f' % (RS_s.predict(X_train).mean(), RS_s.M.alpha))
        _log.info('Coverage of reference points: %.3f' % RS_s.predict(RS_s.refSamples).mean())

    fir = utils.eval_false_inclusion_rate(RS_s, check_incl)
    ex.info['false_inclusion_rate'] = fir
    ex.info['data_coverage'] = float(RS_s.predict(X_train).mean())
    ex.info['reference_coverage'] = float(RS_s.predict(RS_s.refSamples).mean())

    rules = RS_s.rules(transform=lambda a,b: b, fmt='%.1f')
    ex.info['n_rules'] = float(len(rules))
    ex.info['n_rules_literals'] = float(np.sum([len(rule) for rule in rules]))
    ex.info['rules'] = rules

    # Record more detailed rules information, e.g., proportion covered
    D = pd.concat([
        X_train,
        pd.DataFrame(np.ones_like(Y_train), columns=['support_set'])
        ], axis=1)
    Cs = utils.compliance(D, rules)
    # This is everywhere, to be clear
    I1 = np.where(D['support_set'].values==1)[0]

    rule_stats = []
    for i in range(len(rules)):
        # Instances covered by rule
        d = {}
        d['rule'] = rules[i]
        d['n_covered'] = float(Cs[i][:,I1].prod(0).sum())
        d['p_covered'] = float(Cs[i][:,I1].prod(0).mean())
        rule_stats.append(d)

    ex.info['rule_stats'] = rule_stats
    _log.info("Experiment Complete")

    return fir
