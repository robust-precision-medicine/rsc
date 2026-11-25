import numpy as np
import pandas as pd

def pseudo_bulk(adata, individual_label, condition_label, covar_labels, raw_layer = None, return_n_cells = False, mixed_covar_labels = []):
    covar_cond_labels = [i for i in covar_labels]+[condition_label]
    ind_ids = np.unique(adata.obs[individual_label])
    counts = np.zeros((len(ind_ids),adata.shape[1]))
    design_df = pd.DataFrame(np.nan, index=ind_ids, columns=covar_cond_labels)
    n_cells = np.zeros((len(ind_ids,)))
    for i in range(len(ind_ids)):
        n_cells[i] = np.sum(adata.obs[individual_label] == ind_ids[i])
        if raw_layer is not None:
            counts[i,:] = adata[adata.obs[individual_label] == ind_ids[i],:].layers[raw_layer].sum(axis=0)
        else:
            counts[i,:] = adata[adata.obs[individual_label] == ind_ids[i],:].X.sum(axis=0)
        design_df.loc[ind_ids[i],condition_label] = np.unique(adata.obs.loc[adata.obs[individual_label] == ind_ids[i], condition_label])[0]
        for covar_label in covar_labels:
            unq = np.unique(adata.obs.loc[adata.obs[individual_label] == ind_ids[i], covar_label])
            # mixed_covar_labels indicates binary covariates that may assign both 0 and 1 to cells from the same sample; otherwise there should only be a single value per sample since these are sample-level covariates
            assert (len(unq) == 1) or (len(unq)>=2 and (covar_label in mixed_covar_labels))
            design_df.loc[ind_ids[i],covar_label] = np.max(unq)
    counts_df = pd.DataFrame(data=counts, index=ind_ids, columns=adata.var.index)
    dtypes = adata.obs[covar_cond_labels].dtypes
    for i in range(len(dtypes)):
        design_df.iloc[:,i] = design_df.iloc[:,i].astype(dtypes[i])
    if return_n_cells:
        n_cells = pd.DataFrame(data={"n_cells":n_cells}, index=ind_ids)
        return counts_df, design_df, n_cells
    else:
        return counts_df, design_df