conda create --name overrule python=3.6
source activate overrule

conda install numpy
conda install pandas
conda install scipy
conda install scikit-learn
conda install matplotlib
conda install python-graphviz
pip install cvxpy==1.0.21

# Used for running MaxBox baseline
conda install rpy2

# For viewing the experiment notebooks
conda install jupyter

# These are experiment logging utils for the supplement experiments
pip install sacred==0.7.5
pip install tinydb==3.14.1
pip install tinydb-serialization==1.0.4
pip install hashfs
conda install pymongo # Dependency of Sacred, but not used
conda install tqdm
