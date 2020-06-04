#!/usr/bin/env python
# coding: utf-8

from sacred.observers import TinyDbReader

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_exclusion_metadata(
    d, 
    ideal_rule = [['i0', 'not', ''], ['i1', 'not', '']], 
    w_lb=1e-8):
    
    r = dict(d)
    
    r['rule_avg_coverage'] = np.mean([rule['p_covered'] for rule in r['rule_stats']])
    r['rule_n_perfect'] = np.sum([rule['n_covered'] == 0 for rule in r['rule_stats']])
    r['rule_n_total'] = len(r['rule_stats'])
    r['rule_avg_length'] = np.mean([len(rule) for rule in r['rules']])
    
    ideal_literal_idx = np.array([i in ideal_rule for i in r['z_index']])
    dirty_rules_idx = r['z_values'][ideal_literal_idx].sum(axis=0) == len(ideal_rule)
    clean_rules_idx = np.logical_and(
        dirty_rules_idx,
        r['z_values'][~ideal_literal_idx].sum(axis=0) == 0
    )

    # Make sure dirty rules exclude the clean rule
    dirty_rules_idx = np.logical_xor(dirty_rules_idx, clean_rules_idx)
    other_rules_idx = np.logical_not(np.logical_or(dirty_rules_idx, clean_rules_idx))

    assert sum(clean_rules_idx) <= 1
        
    # Rules considered (i.e., they show up in W at all)
    r['n_lp_rules_considered_dirty'] = dirty_rules_idx.sum()
    r['n_lp_rules_considered_clean'] = clean_rules_idx.sum()
    r['n_lp_rules_considered_other'] = other_rules_idx.sum()

    # Rules used (i.e., non-zero values in W)
    r['n_lp_coeff_above_lb_dirty'] = np.logical_and(
        dirty_rules_idx, r['w'] > w_lb).sum()
    r['n_lp_coeff_above_lb_clean'] = np.logical_and(
        clean_rules_idx, r['w'] > w_lb).sum()
    r['n_lp_coeff_above_lb_other'] = np.logical_and(
        other_rules_idx, r['w'] > w_lb).sum()    
    
    # Average value of coefficients
#    r['lp_coeff_avg_value_dirty'] = np.nan if dirty_rules_idx.sum() == 0 else np.mean(r['w'][dirty_rules_idx])
#    r['lp_coeff_avg_value_clean'] = np.nan if clean_rules_idx.sum() == 0 else np.mean(r['w'][clean_rules_idx])
#    r['lp_coeff_avg_value_other'] = np.nan if other_rules_idx.sum() == 0 else np.mean(r['w'][other_rules_idx])
    
    r['n_rounded_rules_considered_clean'] = sum(this_r == ideal_rule for this_r in r['rules'])
    r['n_rounded_rules_considered_dirty'] = \
        sum([np.all(np.array([i in ideal_rule for i in this_r])) for this_r in r['rules']]) - \
        sum(this_r == ideal_rule for this_r in r['rules'])
    
    r['n_lp_rules_viewed'] = r['z_values'].shape[1]
    del r['rules']
    del r['w']
    del r['z_index']
    del r['z_values']
    del r['rule_stats']
    
    return r

def rename_filter_df(df):
    return df.rename(columns={'n_rounded_rules_considered_clean': 'id_exclusion_rr',
                        'n_lp_rules_considered_clean' : 'id_exclusion_lp',
                        'reference_coverage': 'ref_coverage',
                        'literals': 'n_rules_literals'})[['B', 'K', 'alpha', 'lambda0', 'lambda1', 'n_ref_mult',
                                      'lp_obj', 'rounded_obj', 'ref_coverage',
                                      'n_lp_rules_viewed', 'id_exclusion_lp', 'id_exclusion_rr',
                                      'n_rules', 'rule_n_perfect', 'rule_avg_coverage', 'rule_avg_length']]


def get_data(data_path, verbose=False):
    reader = TinyDbReader(data_path)
    meta = reader.fetch_metadata(exp_name='synthetic_removal')
    if verbose:
        print("{} / {} experiments completed".format(
            len([d['status'] for d in meta if d['status'] == 'COMPLETED']),
            len([d['status'] for d in meta])))

    info = [d['info'] for d in meta if d['status'] == 'COMPLETED']
    data = [get_exclusion_metadata(d) for d in info]

    df = rename_filter_df(pd.DataFrame(data))
    return df, info
