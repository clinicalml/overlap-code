#!/bin/sh

# You may need to change the path
# cd $HOME/overlap-code/exps/supp-synthetic
RESULT_DIR="results/grid_search"

for rep in {1..3}
do
  for outer_args in "nr=10 nn=10 K=10 COVERAGE=False B="{10,15,20,25,30}" LAMBDA1="{1e-2,1e-4,1e-6}
  do
    wait
    for inner_args in "LAMBDA0="{1e-2,1e-4,1e-6,0}" ALPHA="{0.95,0.96,0.97,0.98,0.99}
      do
        echo `date +"%Y-%m-%d %T"` "with SEED=${rep} ${outer_args} ${inner_args}"
        THIS_DIR=${RESULT_DIR}_${inner_args// /_}
        python run_experiment.py -t ${THIS_DIR} -l INFO with SEED=$rep $outer_args $inner_args &
      done
  done
done
