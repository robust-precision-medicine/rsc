# rsc

Robust single-cell (rsc) differential expression and differential composition analysis for single-cell RNA-seq data based on subsampling.

Version 1.0.0

The following environment was tested:
```
conda create -n robust_sc python=3.9
conda activate robust_sc
pip install scanpy
pip install statsmodels
conda install -c conda-forge multiprocess
```

To run the tests:
```
cd tests
python -m unittest -f test_rsc
```

Example:
```
import sys
sys.path.insert(1, "path_to_rsc") # the folder with rsc.py, rsc_models.py, rsc_utils.py and logger.py
from rsc import rsc

# adata is an anndata object
rand_seed = 100 # some number to be used as random seed for reproducibility
# init constructor
robsc = rsc(adata, # the anndata
  individual_label="donor_id", # the name of the column in adata.obs that maps to individual identifier
  condition_label="y", # the name of the column in adata.obs that maps to the binary condition (e.g., 0 for control and 1 for case)
  subset_size = 2,
  n_subsets=100,
  rand_seed = rand_seed
  )
q = robsc.estimate_q(N = 10, # num of permutations
  statistical_test="ttest", # for differential composition, "chi" was evaluated (chi-squared)
  verbose = True,
  rand_seed = rand_seed)
df = robsc.subset_analysis(q,
  statistical_test = "ttest",
  sort_results=False,
  verbose = True,
  rand_seed = rand_seed)
```
