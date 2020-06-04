# Running Synthetic Experiments from the Supplement

These experiments make use of `sacred` to track experimental results, using the `tinydb` observer.  See [sacred documentation](https://sacred.readthedocs.io/en/stable/observers.html#tinydb-observer) for more information.  The installation of these packages is covered in `setup.sh` in the root directory of this repo.

The below instructions assume that you are **currently in the `supp-synthetic` directory**, and should be executed as written:  If e.g., you `cd` into the `bash_scripts` folder to run the script, the paths will NOT be correct.

1. Run `./bash_scripts/grid_search_all.sh` to generate data and run full hyperparameter search.  This runs multiple experiments in parallel, but can still take a significant amount of time.
  + Note: This is a wrapper for `run_experiment.py`, which is called once for every hyperparameter configuration
  + Results are stored in various folders in `./results` 
2. `./notebooks/supplement-A-synthetic.ipynb` reads in those results and creates the relevant tables that appear in the supplement
