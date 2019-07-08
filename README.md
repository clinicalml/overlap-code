# OverRule: Overlap Estimation using Rule Sets

## Prerequisites

OverRule is built on Python 3 with Pandas, numpy, scikit-learn and cvxpy.

## Code example

The script `setup.sh` assumes that `anaconda` is installed, and creates a virtual environment names `overrule` for the relevant packages.

Once this is done, the shell script `run_iris_exp.sh` runs OverRule on the Iris dataset and stores the output in the folder `./results`.

## Acknowledgements

The script `maxbox.R` is due to Colin B. Fogarty from http://www.mit.edu/~cfogarty/maxbox.R, and is used to replicate MaxBox as a baseline method
