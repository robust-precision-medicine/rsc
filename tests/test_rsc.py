# run tests as follows (-f for stopping if fails in any test):
# python -m unittest -f test_rsc

import unittest
import numpy as np
import pandas as pd
import scanpy as sc
import sys
sys.path.insert(1, '../')
from rsc import *
import rsc_models

class test_rsc(unittest.TestCase):

	# sample data for testing	
	SAMPLE_DATA_PATH = "sample_data.h5ad"
	adata = sc.read_h5ad(SAMPLE_DATA_PATH)
	adata.obs["y"] = pd.get_dummies(adata.obs["sex"]).iloc[:, 0]
	adata_hvg = adata[:,np.argsort(-np.ravel(np.argsort(-adata.X.todense().var(axis=0))))[0:100]].copy()

	def test_integration(self):
		# integration test
		rand_seed = 100
		for statistical_test in rsc.DE_MODELS.keys():
			robsc = rsc(self.adata_hvg, individual_label="donor_id", condition_label="y",
			lib_size_label = "nCount_RNA", covar_labels = [],
			subset_size = 1, n_subsets=10, rand_seed = rand_seed)
			q = robsc.estimate_q(N = 2, statistical_test=statistical_test, verbose = False, rand_seed = rand_seed)
			self.assertTrue(len(q) == self.adata_hvg.shape[1])
			df = robsc.subset_analysis(q, statistical_test = statistical_test, sort_results=False, verbose = False, rand_seed = rand_seed)
			self.assertTrue(df.shape[0] == self.adata_hvg.shape[1])

if __name__ == '__main__':
	unittest.main()