# OverRule: Overlap Estimation using Rule Sets

## Prerequisites

OverRule is built on Python 3 with Pandas, numpy, scikit-learn and cvxpy.

The script `setup.sh` assumes that `anaconda` is installed, and creates a virtual environment named `overrule` for the relevant packages.  It is divided into two sections:  The first installs the minimum dependencies for `overrule` to run, and the second install dependencies required to reproducing results end-to-end (including e.g., `jupyter` and `sacred` for logging experiment results)

## Reproducing Synthetic Experiments

Once `setup.sh` has been run, the Jupyter Notebook `./exps/iris/exp_iris_2d.ipynb` runs OverRule on the Iris dataset and stores the output in the folder `./exps/iris/results`.  This reproduces Figure 2 in the main paper.

Similarly, `./exps/supp-synthetic/README.md` gives details on how to reproduce the purely synthetic experiments in the supplement.  This reproduces Tables S1, S2, and S5 in the supplement.

## Acknowledgements

The script `maxbox.R` is due to Colin B. Fogarty from http://www.mit.edu/~cfogarty/maxbox.R, and is used to replicate MaxBox as a baseline method
