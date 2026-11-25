from anndata._core.anndata import AnnData
from scipy import stats
import numpy as np
import scanpy as sc
import pandas as pd
import statsmodels.api as sm 
from rsc_utils import *
import scipy

def t_test(adata: AnnData, y, **kwargs):
    assert np.all(np.sort(np.unique(y)) == np.array([0,1]))
    X1_indices = y==1
    X2_indices = y==0
    X1 = adata.X[X1_indices,].todense()
    X2 = adata.X[X2_indices,].todense()
    statistics, pvalues = stats.ttest_ind(X1, X2, axis=0)
    means_diff = (np.mean(X1, axis=0)-np.mean(X2, axis=0))
    sd_pooled = np.sqrt(((np.sum(X1_indices)-1)*np.var(X1, axis=0) + (np.sum(X2_indices)-1)*np.var(X2, axis=0))/(adata.shape[0]-2))
    effect_sizes = np.ravel(means_diff / sd_pooled) # Cohen's d
    return statistics, pvalues, effect_sizes

def chi(adata: AnnData, y, **kwargs):
    assert np.all(np.sort(np.unique(y)) == np.array([0,1]))
    X1_indices = y==1
    X2_indices = y==0
    
    if scipy.sparse.issparse(adata.X):
        X1 = adata.X[X1_indices,].todense()
        X2 = adata.X[X2_indices,].todense()
    else:
        X1 = adata.X[X1_indices,]
        X2 = adata.X[X2_indices,]
    pvalues, statistics, effect_sizes = [], [], []
    celltypes = list(adata.var_names)
    for col, ct in enumerate(celltypes):
        num_ct_X1, num_ct_X2 = np.sum(X1[:, col]), np.sum(X2[:, col])
        totalNum_X1, totalNum_X2 = np.sum(X1, axis=None), np.sum(X2, axis=None)
        table = [[num_ct_X1, num_ct_X2],
                 [totalNum_X1 - num_ct_X1, totalNum_X2 - num_ct_X2]]
        #print(table)
        if np.min(np.array(table)) < 1: # Can't do Chi-Squared test with a 0 element
            chi2, p = 0, 1
        elif np.min(np.array(table)) <= 5: # Apply Yates' correction
            chi2, p, dof, ex = stats.chi2_contingency(table, correction=True)
        else:
            chi2, p, dof, ex = stats.chi2_contingency(table)
        pvalues.append(p)
        statistics.append(chi2)
        phi = np.sqrt(chi2/(totalNum_X1 + totalNum_X2))
        direction = 1
        if num_ct_X1/totalNum_X1 < num_ct_X2/totalNum_X2:
            direction = -1
        effect_sizes.append(phi * direction)
    return statistics, pvalues, effect_sizes

def scanpy_t_test_overestim_var(adata: AnnData, y, **kwargs):
    assert np.all(np.sort(np.unique(y)) == np.array([0,1]))
    X1_indices = y==1
    X2_indices = y==0
    if "log1p" in adata.uns:
        adata.uns["log1p"]["base"] = None
    adata.obs["y_str"] = np.array([str(i) for i in y])    
    sc.tl.rank_genes_groups(adata, groupby="y_str", groups = ["1"], use_raw=False, method="t-test_overestim_var")
    # results are sorted by pvalues; use original order
    df = pd.DataFrame(data={"pvals":[i[0] for i in adata.uns["rank_genes_groups"]["pvals"]], \
        "scores":[i[0] for i in adata.uns["rank_genes_groups"]["scores"]]},
        index = [i[0] for i in adata.uns["rank_genes_groups"]["names"]])
    df = df.loc[adata.var.index,:]
    statistics = df["scores"]
    pvalues = df["pvals"]
    const = np.sqrt((np.sum(X1_indices)-np.sum(X1_indices))/(np.sum(X1_indices)*np.sum(X2_indices)))
    effect_sizes = statistics*const # Cohen's d
    return statistics, pvalues, effect_sizes

def wilcoxon(adata, y, **kwargs):
    assert np.all(np.sort(np.unique(y)) == np.array([0,1]))    
    X1_indices = y==1
    X2_indices = y==0
    X1 = adata.X[X1_indices,].todense()
    X2 = adata.X[X2_indices,].todense()    
    results = stats.mannwhitneyu(X1, X2, alternative="two-sided", axis=0, method="auto")
    effect_sizes = results.statistic/(np.sum(X1_indices)*np.sum(X2_indices))-0.5 # aka the common language effect size; removing 0.5 allows to assign negative values for genes that are estimated to be under-regulated with the condition
    results.statistic
    results.pvalue
    return results.statistic, results.pvalue, effect_sizes

def poisson_regression(adata, y, **kwargs):
    if ("covar_labels" in kwargs) and (kwargs["covar_labels"] is not None) \
        and len(kwargs["covar_labels"]):
        covars = adata.obs.loc[:,kwargs["covar_labels"]].to_numpy()
    else:
        covars = np.empty((len(y),0))
    if ("lib_size_label" in kwargs) and (kwargs["lib_size_label"] is not None):
        lib_size = adata.obs[kwargs["lib_size_label"]].to_numpy()
        offest = np.log(lib_size)
    else:
        offest = None
    n_genes = adata.shape[1]
    pvalues = np.ones((n_genes,))
    statistics = np.zeros((n_genes,))
    effect_sizes = np.zeros((n_genes,))
    X = adata.X.todense()
    C = sm.add_constant(np.column_stack((covars,y)))
    for j in range(n_genes):
        try:
            mdl = sm.GLM(X[:,j], C, offset = offest, family=sm.families.Poisson()).fit()
            pvalues[j] = mdl.pvalues[C.shape[1]-1]
            statistics[j] = float(mdl.summary().tables[1].data[C.shape[1]][3])
            effect_sizes[j] = float(mdl.summary().tables[1].data[C.shape[1]][1])
        except ValueError:
            continue
    return statistics, pvalues, effect_sizes


