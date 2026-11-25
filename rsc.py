from typing import Union, Literal, List, Callable, Dict
from anndata._core.anndata import AnnData
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import rsc_models
import multiprocess as mp
import warnings
import time
from scipy.stats import binom

class rsc:
    
    DE_MODELS = {"ttest": rsc_models.t_test,
        "scanpy_t_test_overestim_var": rsc_models.scanpy_t_test_overestim_var,
        "wilcoxon": rsc_models.wilcoxon,
        "poisson_regression": rsc_models.poisson_regression,
        "chi": rsc_models.chi}

    DE_MODELS_REQUIRE_LIB_SIZE = {"ttest": False,
        "scanpy_t_test_overestim_var": False,
        "wilcoxon": False,
        "chi": False,
        "poisson_regression": True}

    MIN_VAR = 1e-12

    def __init__(self,
        adata: AnnData,
        individual_label: str,
        condition_label: str,
        lib_size_label: str = None,
        covar_labels: list[str] = [],
        raw_layer: str = None,
        subset_size: int = 2,
        n_subsets: int = 100,
        rand_seed: int = 1000):
        
        rng = np.random.default_rng(rand_seed)
        
        # input validation
        assert condition_label in adata.obs, "condition_label does not exist in adata.obs"
        assert individual_label in adata.obs, "individual_label does not exist in adata.obs"
        if lib_size_label is not None:
            assert lib_size_label in adata.obs, "lib_size_label is not in adata.obs"
        if len(covar_labels):
            assert np.all(np.array([i in adata.obs for i in covar_labels])), "At least one of the covariate labels in covar_labels is not in adata.obs"

        self.adata = adata
        self.n_genes = self.adata.shape[1]
        self.individual_label = individual_label
        self.condition_label = condition_label
        self.lib_size_label = lib_size_label
        self.covar_labels = covar_labels
        self.raw_layer = raw_layer
        self.subset_size = subset_size
        self.n_subsets = n_subsets

        self.ids = np.unique(self.adata.obs[self.individual_label])
        self.condition_categories = np.unique(adata.obs[condition_label])
        self.n_categories = len(self.condition_categories)
        self.condition_groups = self._get_condition_groups(self.adata.obs[self.condition_label])

        # sample subsets
        self.subsets = self._sample_subsets(self.adata.obs[self.condition_label], rng)

        self.sa_raw_results = None
        self.sa_results = None

        # additional input validation
        assert self.n_categories == 2, "The condition must include two categories."
        assert np.all(self.condition_categories == np.array([0,1])), "The values of condition_label must be encoded as 0 and 1 (only two categories are currently supported)."
        assert self.n_categories*subset_size<len(self.ids), "The number of individuals in the data must be larger than the number of categories in condition_label times the subset size."
        assert np.all(np.array([len(g) for g in self.condition_groups])>= self.subset_size), "The condition has to include at least subset_size individuals per category." 
        
        
    def subset_analysis(self,
        q: list[Union[float, np.ndarray]],
        statistical_test: list[Union[Literal["ttest", "chi", "wilcoxon", "poisson_regression"],
        Callable[[AnnData, str, list[str]], list[Dict,Dict] ]]] = "poisson_regression",
        alpha: float = 0.1,
        n_cells_per_gene: int = 20,
        max_nan_frac = 0.5,
        sort_results: bool = True,
        parallel: bool = False,
        num_cores: int = None,
        verbose = True,
        rand_seed: int = 1000):

        start_time = time.time()
        rng = np.random.default_rng(rand_seed)

        # input validation
        assert isinstance(statistical_test, str) or callable(statistical_test)
        assert (isinstance(q, float) and (q>=0 and q<=1)) or \
            (isinstance(q, np.ndarray) and len(q) == self.n_genes), "q has to be scalar of type float or a numpy.ndarray array of length equal to the number of genes in the data."
        
        if isinstance(q, float):
            q = np.zeros((self.n_genes,))+q

        self.n_cells_per_gene = n_cells_per_gene

        num_cores = self._get_num_cores(parallel, num_cores)
        
        func = self._get_statistical_test(statistical_test)

        # calculate pvalues for all genes in every subset
        def test_subset(s):
            indices_s = self.subsets[s]["subset_indices"]
            condition_s = self.adata.obs.loc[indices_s, self.condition_label].to_numpy()
            # genes that are expressed in a small num of cells in the current subset will not be tested
            gene_indices = np.ravel((self.adata.X[indices_s,:]>0).sum(axis=0)>=self.n_cells_per_gene)
            statistics = np.full((len(gene_indices),), np.nan)
            effect_sizes = np.full((len(gene_indices),), np.nan)
            pvalues = np.full((len(gene_indices),), np.nan)
            statistics_, pvalues_, effect_sizes_ = func(self.adata[indices_s,gene_indices],
                condition_s,
                lib_size_label = self.lib_size_label,
                covar_labels = self.covar_labels,
                condition_label = self.condition_label,
                individual_label = self.individual_label,
                raw_layer = self.raw_layer)
            statistics[gene_indices] = statistics_
            pvalues[gene_indices] = pvalues_
            effect_sizes[gene_indices] = effect_sizes_
            return statistics, pvalues, effect_sizes

        subset_results = []
        if parallel:
            pool = mp.Pool(num_cores)
            results = pool.map(test_subset, range(self.n_subsets))
            pool.close()
            for s in range(self.n_subsets):
                subset_results.append({"statistics": results[s][0], "pvalues": results[s][1], "effect_sizes": results[s][2]})
        else:
            for s in range(self.n_subsets):
                results = test_subset(s)
                subset_results.append({"statistics": results[0], "pvalues": results[1], "effect_sizes": results[2]})

        pvalues = np.ones((self.n_genes,))
        pvalues_binom = np.ones((self.n_genes,)) 
        effect_sizes = np.zeros((self.n_genes,))
        effect_sizes_consistency = np.zeros((self.n_genes,)) # the fraction of subsets with pvalue above pvalues_q that agree with the direction of the effect size in effect_sizes
        pvalues_q = np.ones((self.n_genes,)) # quantile q of the pvalues of a gene across all subsets
        nan_fracs = np.zeros((self.n_genes,))
        directions = np.zeros((self.n_genes,))
        for j in range(self.n_genes):
            if np.isnan(q[j]):
               pvalues[j] = np.nan
               effect_sizes[j] = np.nan
               nan_fracs[j] = np.nan
               effect_sizes_consistency[j] = np.nan
            else:
                pvalues_j = np.ones((self.n_subsets,))
                effect_sizes_j = np.zeros((self.n_subsets,))
                
                for s in range(self.n_subsets):
                    pvalues_j[s] = subset_results[s]["pvalues"][j]
                    effect_sizes_j[s] = subset_results[s]["effect_sizes"][j]
                nan_indices_j = np.isnan(pvalues_j)
                nan_fracs[j] = np.sum(nan_indices_j)/self.n_subsets
                if nan_fracs[j] > max_nan_frac:
                    # there are too many subsets with nan pvalue for gene j; digregard this gene from the rest of the analysis
                    pvalues[j] = np.nan
                    pvalues_binom[j] = np.nan 
                    effect_sizes[j] = np.nan
                    effect_sizes_consistency[j] = np.nan
                else:
                    pvalues[j] = self._calc_quantile_pval(pvalues_j[~nan_indices_j], q[j])
                    effect_sizes_j_pool = effect_sizes_j[~nan_indices_j][pvalues_j[~nan_indices_j] >= np.quantile(pvalues_j[~nan_indices_j], q[j], method="closest_observation")]
                    numOver = sum([1 for e in effect_sizes_j_pool if e >=0])
                    numUnder = len(effect_sizes_j_pool) - numOver
                    if numOver >= numUnder:
                        directions[j] = 1
                    else:
                        directions[j] = -1
                    
                    effect_sizes[j] = np.mean(effect_sizes_j[~nan_indices_j])
                    effect_sizes_consistency[j] = np.sum((effect_sizes[j]*effect_sizes_j_pool)>0)/len(effect_sizes_j_pool)
                    
                    if effect_sizes[j]>0:
                        k = np.sum(effect_sizes_j_pool<0)
                    else:
                        k = np.sum(effect_sizes_j_pool>0)
                    pvalues_binom[j] = binom.cdf(k, len(effect_sizes_j_pool), 0.5)  
                    
                    pvalues_q[j] = np.quantile(pvalues_j[~nan_indices_j], q[j], method="closest_observation")
                    
        
        # FDR corrected pvalues at level alpha
        pvalues_fdr_sig = np.full((self.n_genes,), False)
        pvalues_fdr = np.ones((self.n_genes,))
        pvalues_fdr[:] = np.nan
        non_nan_indices = ~np.isnan(pvalues)
        a, b = fdrcorrection(pvalues[non_nan_indices], alpha)
        pvalues_fdr_sig[non_nan_indices] = a
        pvalues_fdr[non_nan_indices] = b

        pvalues_binom_fdr_sig = np.full((self.n_genes,), False)
        pvalues_binom_fdr = np.ones((self.n_genes,))
        pvalues_binom_fdr[:] = np.nan
        non_nan_indices = ~np.isnan(pvalues_binom)
        a, b = fdrcorrection(pvalues_binom[non_nan_indices], alpha)
        pvalues_binom_fdr_sig[non_nan_indices] = a
        pvalues_binom_fdr[non_nan_indices] = b
        
            

        data = {"pvalue": pvalues,
            "pvalue_adjusted": pvalues_fdr,
            "fdr_significant": pvalues_fdr_sig,
            "binom_pvalue": pvalues_binom,
            "binom_pvalue_adjusted": pvalues_binom_fdr,
            "binom_fdr_significant": pvalues_binom_fdr_sig,
            "quantile": q,
            "effect_size": effect_sizes,
            "direction": directions,
            "effect_dir_consistency": effect_sizes_consistency,
            "nan_frac": nan_fracs,
            "pvalue_q": pvalues_q}
        df = pd.DataFrame(data, index = self.adata.var.index)
        
        self.sa_raw_results = subset_results
        self.sa_results = df

        if sort_results:
            df = df.sort_values("pvalue")

        if verbose:
            print("Execution of {} was completed in {}.".format("subset_analysis", self._get_runtime(start_time)))
        return df

    def estimate_q(self,        
        N: int = 20,
        statistical_test: Literal["ttest", "chi", "wilcoxon", "poisson_regression"] = "ttest",
        alpha: float = 0.1,
        n_cells_per_gene: int = 20,
        max_nan_frac = 0.5,
        delta: float = 0.05,
        gamma: float = None,
        parallel: bool = False,
        num_cores: int = None,
        binom_calib: bool = True, # whether to consider binomial p-values in estimating q
        beta_calib: bool = True, # whether to consider beta p-values in estimating q
        verbose: bool = True,
        rand_seed: int = 1000):
        
        assert binom_calib or beta_calib, "at least one of binom_calib, beta_calib has to be set to True"

        start_time = time.time()
        rng = np.random.default_rng(rand_seed)
        
        func = self._get_statistical_test(statistical_test)
        num_cores = self._get_num_cores(parallel, num_cores, N)

        self.n_cells_per_gene = n_cells_per_gene
        
        # define the set of possible quantiles
        Q = np.around(np.cumsum(np.zeros((int(np.ceil(1/delta)))) + delta), decimals=1+int(np.ceil(np.log10(1/delta)))) # note that rounding is critical
        Q = Q[Q<=1]
        
        perm_subsets = []

        # shuffle the condition N times; subset pvalues will be calculated under each of the N permutations
        perm_subsets_iters = []
        perm_conditions = []
        for iter in range(N):
            # shuffle s.t. we preserve the case/control ratio in both "cases" and "controls" (so that we don't have label assignment that is too correlated with the true label)
            assert self.n_categories == 2, "Only two categories are currently supported."
            perm_condition = self.adata.obs[self.condition_label].copy()
            g0 = np.unique(self.adata.obs[self.individual_label][self.adata.obs[self.condition_label]==self.condition_categories[0]])
            g1 = np.unique(self.adata.obs[self.individual_label][self.adata.obs[self.condition_label]==self.condition_categories[1]])
            n_inds_to_move = int(np.round(len(g0)*(1-len(g0)/(len(g0)+len(g1))))) # this is the num of samples to "move" between condition groups in order to keep the same sizes of the two original groups
            a = rng.permutation(g0)[0:n_inds_to_move]
            b = rng.permutation(g1)[0:n_inds_to_move]
            perm_condition[self.adata.obs[self.individual_label].isin(a)] = self.condition_categories[1]
            perm_condition[self.adata.obs[self.individual_label].isin(b)] = self.condition_categories[0]
            # sample subsets
            perm_subsets_iters.append(self._sample_subsets(perm_condition, rng))
            perm_conditions.append(perm_condition)
        
        def run_iter(iter):
            # run statistical testing on all genes in each of the subsets:
            perm_subsets_iter = perm_subsets_iters[iter]
            perm_condition = perm_conditions[iter]
            def test_subset(s):
                indices_s = perm_subsets_iter[s]["subset_indices"]
                perm_condition_s = perm_condition[indices_s].to_numpy()
                # genes that are expressed in a small num of cells in the current subset will not be tested
                gene_indices = np.ravel((self.adata.X[indices_s,:]>0).sum(axis=0)>=self.n_cells_per_gene)
                pvalues = np.full((len(gene_indices),), np.nan)
                effect_sizes = np.full((len(gene_indices),), np.nan)
                _, pvalues_, effect_sizes_ = func(self.adata[indices_s,gene_indices], \
                    perm_condition_s,
                    lib_size_label = self.lib_size_label,
                    covar_labels = self.covar_labels,
                    condition_label = self.condition_label,
                    individual_label = self.individual_label,
                    raw_layer = self.raw_layer)
                pvalues[gene_indices] = pvalues_
                effect_sizes[gene_indices] = effect_sizes_
                return pvalues, effect_sizes
            # calculate pvalues for all genes in every subset
            for s in range(self.n_subsets):
                pvalues, effect_sizes = test_subset(s)
                perm_subsets_iter[s]["pvalues"] = pvalues
                perm_subsets_iter[s]["effect_sizes"] = effect_sizes
            return perm_subsets_iter
        if parallel:
            pool = mp.Pool(num_cores)
            results = pool.map(run_iter, range(N))
            pool.close()
            for iter in range(N):
                perm_subsets.append(results[iter])
        else:
            for iter in range(N):
                if verbose:
                    print("Running permutation {} out of {} permutations...".format(iter+1, N))
                perm_subsets.append(run_iter(iter))

        decimals = len(str(delta))-1    # for rounding the selected quantiles

        if verbose:
            print("Calculating p-values for each possible quantile...")
        P_beta_tilde, P_binom_tilde = self._calc_quantile_pvalues(perm_subsets, N, Q, max_nan_frac)

        if verbose:
            print("Initializing quantiles...")
        # Initialize the quantiles q_j for each gene j by finding for each gene the min quantile q with min pval above gamma
        gamma = alpha/(self.n_genes*N)
        min_P_tilde = {} # for storing min pvalue per gene across all N iterations 
        for q in range(len(Q)):
            if beta_calib:
                #print(P_beta_tilde[Q[q]])
                min_P_tilde[Q[q]] = np.amin(P_beta_tilde[Q[q]], axis=0)
            else:
                min_P_tilde[Q[q]] = np.amin(P_binom_tilde[Q[q]], axis=0)
        qjs_init = np.ones((self.n_genes,))
        for j in range(self.n_genes):
            if min_P_tilde[Q[-1]][j] <= gamma:
                # condition is not satisfied by any quantile; assign nan
                qjs_init[j] = np.nan
            else:
                for q in Q:
                    if min_P_tilde[q][j] > gamma:
                        qjs_init[j] = q
                        break
        # enforce p-value calibration by increasing the quantiles qjs until reaching calibration
        if binom_calib:
            if verbose:
                print("Calibration based on Binomial-derived p-values...")
            # calibrate based on the binomial-derived pvalues
            qjs = self._calibrate_pvalues(P_binom_tilde, qjs_init, N, alpha, delta, decimals)
        else:
            qjs = qjs_init
        if beta_calib:
            if verbose:
                print("Calibration based on Beta-derived p-values...")
            # calibrate based on the beta-derived pvalues
            qjs = self._calibrate_pvalues(P_beta_tilde, qjs, N, alpha, delta, decimals)
        
        if verbose:
            print("Execution of {} was completed in {}.".format("estimate_q", self._get_runtime(start_time)))
        return qjs


    def _calc_quantile_pvalues(self, perm_subsets, N, Q, max_nan_frac):
        # calculate a pvalue for each gene under every quantile in Q using the beta distribution
        P_tilde = {}
        P_binom_tilde = {}
        for q in Q:
            P_tilde[q] = np.zeros((N,self.n_genes))
            P_binom_tilde[q] = np.zeros((N,self.n_genes))
        for iter in range(N):
            P_tilde_iter = np.zeros((self.n_subsets,self.n_genes))
            B_tilde_iter = np.zeros((self.n_subsets,self.n_genes))
            for s in range(self.n_subsets):
                P_tilde_iter[s,:] = perm_subsets[iter][s]["pvalues"]
                B_tilde_iter[s,:] = perm_subsets[iter][s]["effect_sizes"]
            for q in Q:
                for j in range(self.n_genes):
                    P_tilde_iter_j = P_tilde_iter[:,j]
                    nan_indices_j = np.isnan(P_tilde_iter_j)
                    if np.sum(nan_indices_j)/self.n_subsets > max_nan_frac:
                        # there are too many subsets with nan pvalue for gene j in iteration iter; disregard this gene
                        P_tilde[q][iter,j] = np.nan
                    else:
                        # calc pvalue based on Beta dist
                        P_tilde[q][iter,j] = self._calc_quantile_pval(P_tilde_iter_j[~nan_indices_j], q)
                        # calc pvalue based on Binomial dist
                        indices = P_tilde_iter_j>=np.quantile(P_tilde_iter_j[~nan_indices_j], q, method="closest_observation") # consider only subsets with pvalues higher than the quantile q pvalue
                        B_tilde_iter_j = B_tilde_iter[indices,j]
                        a = np.sum(B_tilde_iter_j>0)
                        b = np.sum(B_tilde_iter_j<0)
                        if a>b:
                            k = b
                        else:
                            k = a
                        P_binom_tilde[q][iter,j] = 2*binom.cdf(k, len(B_tilde_iter_j), 0.5)  # the multiplication by 2 is because we consider the abundant direction and don't speicify a direction under H1 (and due to symmetry)
        return P_tilde, P_binom_tilde
    
    def _calibrate_pvalues(self, P_tilde, qjs_init, N, alpha, delta, decimals):
        n_pvalues = N*self.n_genes
        # pvalues under the currnet qjs
        pvals = np.empty((n_pvalues,))
        pvals[:] = np.nan
        for j in range(self.n_genes):
            i = qjs_init[j]
            if ~np.isnan(i):
                pvals[j*N:(j+1)*N] = [p for p in P_tilde[i][:,j]]
        tau = np.flip(np.linspace(min(0.1,10/n_pvalues),0.95,20)) # quantiles to be considered for enforcing calibration of p-values under the null
        ci_coverage = 1-alpha/len(tau) # correct the confidence intervals for multiple hypotheses
        qjs = [i for i in qjs_init]
        gene_scores = np.zeros((self.n_genes,))
        for j in range(self.n_genes):
            if ~np.isnan(qjs_init[j]):
                p_j = P_tilde[qjs_init[j]][:,j]
                p_j = p_j[~np.isnan(p_j)]
                if len(p_j):
                    gene_scores[j] = -np.sum(np.log(p_j))/len(p_j)
        is_calibrated = False # flag for checking whether the distribution of pvalues is calibrated
        is_calibrated_tau = np.full((len(tau),), False) # used in order to skip evaluations of quantiles that have been already calibrated
        while not is_calibrated:
            j = np.argmax(gene_scores)
            if (gene_scores[j] == 0):
                # qjs is nan for all genes
                break
            qjs[j] = np.around(qjs[j] + delta, decimals=decimals)
            if qjs[j] > 1:
                qjs[j] = np.nan
                gene_scores[j] = 0
            else:
                p_j = P_tilde[qjs[j]][:,j]
                p_j = p_j[~np.isnan(p_j)]
                assert len(p_j)>0                    
                gene_scores[j] = -np.sum(np.log(p_j))/len(p_j)
            # update the pvalues of gene j under its new quantile qjs[j]
            if np.isnan(qjs[j]):
                pvals[j*N:(j+1)*N] = np.nan
            else:
                pvals[j*N:(j+1)*N] = [p for p in P_tilde[qjs[j]][:,j]]
            pvals_no_nans = pvals[~np.isnan(pvals)]
            # check if the distribution of pvalues under the current qjs is calibrated using all the quantiles in tau.
            n_non_nan_pvalues = len(pvals_no_nans)
            if n_non_nan_pvalues == 0:
                print("Returning max quantiles")
                return np.array([1 for _ in qjs_init])
            sorted_pvals = np.sort(pvals_no_nans)
            pvals_quantiles = np.quantile(pvals_no_nans, tau, method="closest_observation")
            is_calibrated = True
            for t in range(len(tau)):
                if is_calibrated_tau[t]: continue # the distribution is already callibrated wrt this quantile
                # find ci_coverage confidence interval for the tau[t]-th quantile of the current pvalues
                pos = np.where(sorted_pvals == pvals_quantiles[t])
                u = np.max(pos)
                l = np.min(pos)
                while (stats.binom.cdf(u, n_non_nan_pvalues, tau[t])-stats.binom.cdf(l, n_non_nan_pvalues, tau[t])) < ci_coverage:
                    if l==0:
                        # reached the min p-value; can only extend the CI from right
                        l = -np.infty
                        u += 1
                    elif u==(n_non_nan_pvalues-1):
                        # reached the max p-value; can only extend the CI from left
                        u = np.infty
                        l -= 1
                    else:
                        u += 1
                        l -= 1
                if l == -np.infty:
                    l = 0
                if sorted_pvals[l] < tau[t]:
                    # results are not calibrated if the tau[t]-th quantile is not higher than the theoretical tau[t] whp
                    is_calibrated = False
                    break
                else:
                    is_calibrated_tau[t] = True
        return np.array([i for i in np.around(qjs, decimals=decimals)])
    
    def _get_num_cores(self,parallel, num_cores, N = None):
        if parallel and (num_cores is None):
            num_cores = max(mp.cpu_count()-1,1)
            if N is not None:
                num_cores = min(N,num_cores)
        return num_cores

    def _get_statistical_test(self, statistical_test):
        if isinstance(statistical_test, str):
            #if self.n_categories > 2:
            #    assert  (statistical_test != "ttest") & (statistical_test != "wilcoxon"), "Cannot perform t-test or Wilcoxon test on condition with more than two categories."
            func = self.DE_MODELS[statistical_test]
            if statistical_test in self.DE_MODELS_REQUIRE_LIB_SIZE:
                if self.DE_MODELS_REQUIRE_LIB_SIZE[statistical_test] and (self.lib_size_label is None):
                    warnings.warn("statistical_test was set to {}, which should account for library size, however, library size was not set during class initialization.".format(statistical_test))
        else:
            func = statistical_test
        return func


    def _get_condition_groups(self, condition):
        groups = []
        for c in self.condition_categories:
            groups.append(np.unique(self.adata.obs.loc[condition == c, self.individual_label]))
        return groups

    # A function for sampling subsets, each of which includes a randomly selected subset of each of the categories of the condition.
    def _sample_subsets(self, condition, rng):
        groups = self._get_condition_groups(condition)
        subsets = {}
        for num_subset in range(self.n_subsets):
            subset_groups = []
            subset_samples = []
            for g in range(len(groups)):
                subset_groups.append(groups[g][rng.permutation(len(groups[g]))[0:self.subset_size]])
                subset_samples.extend(subset_groups[g])
            subset_indices = self.adata.obs[self.individual_label].isin(subset_samples)
            subsets[num_subset] = {"subset_groups": subset_groups,
                "subset_samples": subset_samples,
                "subset_indices": subset_indices}
        return subsets
    
    def _calc_quantile_pval(self, pvalues, q):
        if q == 1:
            return np.amax(pvalues, axis=0)
        else:
            return stats.beta.cdf(np.quantile(pvalues, q, axis=0, method="closest_observation"), 1, self.n_subsets*(1-q))
    
    def _get_runtime(self, start_time):
        t = time.time() - start_time
        end_time = t
        if t < 60:
            unit = "seconds"
        elif t >= 60 and t<3600:
            unit = "minutes"
            end_time = end_time/60
        else:
            unit = "hours"
            end_time = end_time/3600
        return "{:.2f} {}".format(end_time, unit)
