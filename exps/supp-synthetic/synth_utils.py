import pandas as pd
import numpy as np

identity_func = lambda a, b: b

def compliance(D, R, inv_trans=lambda x,y : y):
    ops = {'<=': (lambda x,y : x <= y),
          '>': (lambda x,y : x > y),
          '>=': (lambda x,y : x >= y),
          '<': (lambda x,y : x < y),
          '==': (lambda x,y : x == y),
          '': (lambda x,y : x==True),
          'not': (lambda x,y : x==False)}

    Ws = []
    for r in R:
        W = []
        for c in r:
            try:
                v = float(c[2])
            except:
                v = c[2]
            W.append(ops[c[1]](inv_trans(c[0], D[c[0]].values), v))
        W = np.array(W)
        Ws.append(W)
    return Ws

def calc_coverage(X, RS_s):
    # This predicts whether or not X trips ANY of the CONSIDERED rules
    x_by_all_rules = RS_s.predict_rules(X)
    # This lays out the set of singletons* x CONSIDERED rules
    # *this is 2 x dimension for binary variables
    clauses_by_all_rules = RS_s.M.z
    # This lays out the FINAL rules with 1,0, after rounding
    rules_used_idx = RS_s.M.w == 1

    x_by_used_rules = x_by_all_rules[:, rules_used_idx]
    prop_covered_by_used_rule = x_by_used_rules.mean(axis=0)

    return prop_covered_by_used_rule

def eval_confusion_matrix(RS_s, x, check_fn):
    # Predicted reference samples in support
    pred_ref = RS_s.predict(RS_s.refSamples)
    # Actual reference samples in support
    true_ref = check_fn(RS_s.refSamples)
    # Check the confusion matrix
    ct_ref = pd.crosstab(
            pred_ref, true_ref,
            rownames=['Predicted'], colnames=['Actual'])

    # Predicted reference samples in support
    pred_dat = RS_s.predict(x)
    true_dat = check_fn(x)
    ct_dat = pd.crosstab(pred_dat, true_dat,
            rownames=['Predicted'], colnames=['Actual'])

    return cmat_ref, cmat_X

def eval_false_inclusion_rate(RS_s, check_fn):
    # Predicted reference samples in support
    pred_ref = RS_s.predict(RS_s.refSamples)
    # Actual reference samples in support
    true_ref = check_fn(RS_s.refSamples)

    # Of the reference samples that should be excluded, how many get through?
    false_inclusion_rate = pred_ref[true_ref == 0].mean()

    return false_inclusion_rate

    return cmat_ref, cmat_X

def print_synth_rules(X, RS_s, CNF=True):
    rules_support = RS_s.rules(transform=identity_func, fmt='%.1f')
    prop_covered = calc_coverage(X, RS_s)

    # Outer logic takes into account the negation of the CNF
    outer_logic = ['NOT', 'AND NOT'] if CNF else ['   ', 'AND']
    inner_logic = ['   ', 'AND'] if CNF else ['EITHER', 'OR']

    print("Total coverage of X: {:.3f}".format(RS_s.predict(X).mean()))
    print("Total volume:        {:.3f}".format(RS_s.predict(RS_s.refSamples).mean()))
    print("-----------")
    for i in range(len(rules_support)):
        this_rule = rules_support[i]
        print("{:<7} Rule: {} \t \t \t Covers {:.3f} of X".format(
            outer_logic[0] if i == 0 else outer_logic[1], 
            i,
            prop_covered[i]))
        print("{")
        for j in range(len(this_rule)):
            this_clause = this_rule[j]
            print("\t {} {} {}".format(
                inner_logic[0] if j == 0 else inner_logic[1],
                'NOT' if this_clause[1] == 'not' else '   ', 
                this_clause[0]))
        print("}")
