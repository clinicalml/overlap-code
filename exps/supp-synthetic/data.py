#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Goal: Create a synthetic environment in which it is particlarly challenging for the
# algorithm to find a lack of support

import numpy as np
import pandas as pd

def get_feat(n_samps, n_pos_samps):
    # Get a feature with a fixed marginal distribution
    this_feat = np.zeros((n_samps, 1))
    this_feat[:n_pos_samps] = 1
    return np.random.permutation(this_feat)

def advCube(ns=10000, nd=10, nr=2, ni=2, rare_prob=0.01, seed=2, CNF=True,
        fix_marginal=False):
    """ Generates data from an adversarial hypercube, indexed by some parameters
    which indicate difficulty (TBD)

    args:
        nd: Number of dimensions
        nr: Number of dimensions with a rare binary variable
        r_prob: Probability of rare feature
        ni: Number of dimensions required to capture the missing piece, e.g.,
            the density in dimensions (x_1 = 0 & ... & x_i = 0) is zero
        ns: Number of samples (PRIOR to removal!)
        seed: Random seed
        CNF: Removes AND*(interactions) if true, otherwise OR*(interactions)
        fix_marginal: Fixes the marginal proportions

    returns:
        x: Sample features
    """
    np.random.seed(seed)
    assert ni + nr < nd

    if fix_marginal:
        # Get fixed marginals
        x = np.zeros((ns, nd))
        for i in range(nd):
            this_prob = rare_prob if i < nr else 0.5
            x[:, [i]] = get_feat(ns, int(this_prob * ns))

    else:
        # Sample X from the hypercube 

        # Calculate feature probabilities
        rare_feat_probs = np.repeat(rare_prob, nr)
        other_feat_probs = np.repeat(0.5, nd - nr)

        # Note that the rare probabilities come last
        feat_probs = np.concatenate([other_feat_probs, rare_feat_probs])
        x = np.random.binomial(1, feat_probs, size=(ns, nd))

    collist = ['i{}'.format(i) for i in range(ni)] + \
              ['n{}'.format(i) for i in range(nd - nr - ni)] + \
              ['r{}'.format(i) for i in range(nr)]
    x = pd.DataFrame(x, columns=collist)

    # Remove the required dimension
    # First we have to find those which we need to remove!
    if CNF:
        remove_x = np.all(x.iloc[:, :ni] == 0, axis=1)
        prop_remove = remove_x.mean()

        check_incl = lambda x: np.logical_not(
                #np.logical_or(
                # These fail the synthetic exclusion rule
                np.all(x.iloc[:, :ni] == 0, axis=1)#,
                # This says if any of the rare features are on
                # np.any(x.iloc[:, -nr:] == 1, axis=1)
                )#)
    else:
        remove_x = np.any(x.iloc[:, :ni] == 0, axis=1)
        prop_remove = remove_x.mean()
        check_incl = lambda x: np.logical_not(
                #np.logical_or(
                # These fail the synthetic exclusion rule
                np.any(x.iloc[:, :ni] == 0, axis=1)#,
                # This says if any of the rare features are on
                # np.any(x.iloc[:, -nr:] == 1, axis=1)
                )#)

    return x[~remove_x], check_incl
