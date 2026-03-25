// aipw_fast.cpp — Optimized AIPW scorer with precomputed forest cache
//
// Two-phase design:
//   Phase 1 (precompute_forest_cache_cpp): Called once per honesty split.
//     Flattens tree structure into contiguous arrays, routes honest obs to
//     leaves, computes leaf Y-means. O(n_hon * B * depth) but done once.
//   Phase 2 (aipw_scores_cached_cpp): Called per variable.
//     Only counterfactual routing + phi computation. Reads from cache.
//     No tree extraction, no obs_leaf routing, no leaf mean computation.
//
// Memory layout: all trees concatenated with offset index for cache-friendly
// sequential access. X stored as row-major for locality during tree walks.

#include <Rcpp.h>
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

static const double LEAF_EMPTY = -1e308;


// ============================================================
// Phase 1: Precompute forest cache
// ============================================================

// [[Rcpp::export]]
List precompute_forest_cache_cpp(
    List forest, NumericMatrix X_obs, NumericVector y_honest,
    IntegerVector honest_idx
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_obs.nrow();
    int p = X_obs.ncol();
    int n_hon = honest_idx.size();
    const double* Xobs_ptr = REAL(X_obs);
    const double* y_ptr = REAL(y_honest);

    // --- Flatten tree structure into contiguous arrays ---
    // First pass: count total nodes
    int total_nodes = 0;
    IntegerVector tree_offsets(B + 1);
    IntegerVector tree_nnodes(B);
    for (int b = 0; b < B; b++) {
        IntegerVector sv_r = svl[b];
        tree_offsets[b] = total_nodes;
        tree_nnodes[b] = sv_r.size();
        total_nodes += sv_r.size();
    }
    tree_offsets[B] = total_nodes;

    // Second pass: fill flat arrays
    IntegerVector flat_sv(total_nodes);
    NumericVector flat_sval(total_nodes);
    IntegerVector flat_lc(total_nodes);
    IntegerVector flat_rc(total_nodes);

    for (int b = 0; b < B; b++) {
        IntegerVector sv_r = svl[b];
        NumericVector sval_r = svall[b];
        List ch = chl[b];
        IntegerVector lc_r = ch[0], rc_r = ch[1];
        int off = tree_offsets[b];
        int nn = tree_nnodes[b];
        for (int nd = 0; nd < nn; nd++) {
            flat_sv[off + nd] = sv_r[nd];
            flat_sval[off + nd] = sval_r[nd];
            flat_lc[off + nd] = lc_r[nd];
            flat_rc[off + nd] = rc_r[nd];
        }
    }

    // --- Build row-major X for cache-friendly tree walks ---
    // Flat vector: X_row_flat[j * p + c] = X_obs[honest_idx[j], c]
    // Each observation's features are contiguous in memory.
    NumericVector X_row_flat(n_hon * p);
    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;
        int base = j * p;
        for (int c = 0; c < p; c++) {
            X_row_flat[base + c] = Xobs_ptr[i + n * c];
        }
    }

    // --- Route honest obs to leaves, compute leaf means ---
    // obs_leaf_flat: B * n_hon, stored as [tree0_obs0, tree0_obs1, ..., tree1_obs0, ...]
    IntegerVector obs_leaf_flat(B * n_hon);

    // leaf_mean_flat: same layout as flat tree arrays, one mean per node
    NumericVector leaf_mean_flat(total_nodes, LEAF_EMPTY);

    // Temporary per-tree accumulators
    std::vector<double> leaf_ysum;
    std::vector<int> leaf_ycnt;

    for (int b = 0; b < B; b++) {
        int off = tree_offsets[b];
        int nn = tree_nnodes[b];
        const int* sv = INTEGER(flat_sv) + off;
        const double* sval = REAL(flat_sval) + off;
        const int* lc = INTEGER(flat_lc) + off;
        const int* rc = INTEGER(flat_rc) + off;

        leaf_ysum.assign(nn, 0.0);
        leaf_ycnt.assign(nn, 0);

        int leaf_off = b * n_hon;
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double yi = y_ptr[i];
            // Tree walk using row-major X
            int node = 0;
            int xbase = j * p;
            while (lc[node] != 0 || rc[node] != 0) {
                node = (X_row_flat[xbase + sv[node]] <= sval[node]) ? lc[node] : rc[node];
            }
            obs_leaf_flat[leaf_off + j] = node;
            if (!ISNA(yi)) {
                leaf_ysum[node] += yi;
                leaf_ycnt[node]++;
            }
        }

        // Compute leaf means
        for (int nd = 0; nd < nn; nd++) {
            if (leaf_ycnt[nd] > 0) {
                leaf_mean_flat[off + nd] = leaf_ysum[nd] / leaf_ycnt[nd];
            }
        }
    }

    // --- Which trees split on each variable? ---
    // Store as a p x B boolean matrix (flat), so per-variable lookup is O(1)
    IntegerVector splits_on(p * B, 0);
    for (int b = 0; b < B; b++) {
        int off = tree_offsets[b];
        int nn = tree_nnodes[b];
        for (int nd = 0; nd < nn; nd++) {
            if (flat_lc[off + nd] != 0 && flat_sv[off + nd] < p) {
                splits_on[flat_sv[off + nd] * B + b] = 1;
            }
        }
    }

    // --- Precompute fhat_obs: mean prediction at actual covariates ---
    // Variable-independent: same for every effect() call.
    // fhat_obs[j] = mean of lm[obs_leaf] across all B trees for honest obs j
    NumericVector fhat_obs(n_hon);
    for (int j = 0; j < n_hon; j++) {
        double sum = 0.0; int cnt = 0;
        for (int b = 0; b < B; b++) {
            int off = tree_offsets[b];
            int leaf = obs_leaf_flat[b * n_hon + j];
            double v = REAL(leaf_mean_flat)[off + leaf];
            if (v != LEAF_EMPTY) { sum += v; cnt++; }
        }
        fhat_obs[j] = (cnt > 0) ? sum / cnt : NA_REAL;
    }

    return List::create(
        Named("B") = B,
        Named("n") = n,
        Named("p") = p,
        Named("n_hon") = n_hon,
        Named("honest_idx") = honest_idx,
        Named("y_honest") = y_honest,
        Named("tree_offsets") = tree_offsets,
        Named("tree_nnodes") = tree_nnodes,
        Named("flat_sv") = flat_sv,
        Named("flat_sval") = flat_sval,
        Named("flat_lc") = flat_lc,
        Named("flat_rc") = flat_rc,
        Named("X_row_flat") = X_row_flat,
        Named("obs_leaf_flat") = obs_leaf_flat,
        Named("leaf_mean_flat") = leaf_mean_flat,
        Named("splits_on") = splits_on,
        Named("fhat_obs") = fhat_obs
    );
}


// ============================================================
// Phase 2: Fast AIPW scoring using precomputed cache
// ============================================================

// [[Rcpp::export]]
List aipw_scores_cached_cpp(
    List cache,
    NumericVector ghat,
    int var_col, bool is_binary, double a, double b,
    Nullable<NumericVector> indicator_ = R_NilValue,
    int n_threads = 1
) {
    (void)n_threads;
    int B = as<int>(cache["B"]);
    int n = as<int>(cache["n"]); (void)n;
    int p = as<int>(cache["p"]);
    int n_hon = as<int>(cache["n_hon"]);
    IntegerVector honest_idx = as<IntegerVector>(cache["honest_idx"]);
    NumericVector y_honest = as<NumericVector>(cache["y_honest"]);
    IntegerVector tree_offsets = as<IntegerVector>(cache["tree_offsets"]);
    IntegerVector tree_nnodes = as<IntegerVector>(cache["tree_nnodes"]);
    IntegerVector flat_sv = as<IntegerVector>(cache["flat_sv"]);
    NumericVector flat_sval = as<NumericVector>(cache["flat_sval"]);
    IntegerVector flat_lc = as<IntegerVector>(cache["flat_lc"]);
    IntegerVector flat_rc = as<IntegerVector>(cache["flat_rc"]);
    NumericVector X_row_flat = as<NumericVector>(cache["X_row_flat"]);
    const double* X_ptr = REAL(X_row_flat);
    IntegerVector obs_leaf_flat = as<IntegerVector>(cache["obs_leaf_flat"]);
    NumericVector leaf_mean_flat = as<NumericVector>(cache["leaf_mean_flat"]);
    IntegerVector splits_on = as<IntegerVector>(cache["splits_on"]);
    NumericVector fhat_obs_cached = as<NumericVector>(cache["fhat_obs"]);

    const double* g_ptr = REAL(ghat);
    const double* y_ptr = REAL(y_honest);
    const int* off_arr = INTEGER(tree_offsets);
    const int* sv_all = INTEGER(flat_sv);
    const double* sval_all = REAL(flat_sval);
    const int* lc_all = INTEGER(flat_lc);
    const int* rc_all = INTEGER(flat_rc);
    const double* lm_all = REAL(leaf_mean_flat);
    const int* ol_all = INTEGER(obs_leaf_flat);
    const int* sp_all = INTEGER(splits_on);

    bool has_indicator = indicator_.isNotNull();
    NumericVector indicator_vec;
    const double* ind_ptr = nullptr;
    if (has_indicator) {
        indicator_vec = as<NumericVector>(indicator_);
        ind_ptr = REAL(indicator_vec);
    }

    // Precompute sigma2_ej (needed for continuous)
    double sigma2_ej = 0.0;
    if (!is_binary) {
        double ss = 0.0;
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double xj_val = has_indicator ? ind_ptr[i] : X_ptr[j * p + var_col];
            double ej = xj_val - g_ptr[i];
            ss += ej * ej;
        }
        sigma2_ej = ss / n_hon;
    }

    // Count split trees (needed for output)
    int n_split_trees = 0;
    for (int bb = 0; bb < B; bb++)
        if (sp_all[var_col * B + bb] != 0) n_split_trees++;

    // Output arrays
    NumericVector phi(n_hon);
    NumericVector fhat_a_out(n_hon), fhat_b_out(n_hon), fhat_obs_out(n_hon);
    double psi_sum = 0.0; int psi_cnt = 0;
    double sum_pc = 0.0, sum_co = 0.0;
    int n_diff_leaves = 0;


    // Parallel over observations — each observation is independent
    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;
        int xbase = j * p;
        double fa_sum = 0.0, fb_sum = 0.0;
        int fa_cnt = 0, fb_cnt = 0;

        for (int bb = 0; bb < B; bb++) {
            int off = off_arr[bb];
            const int* sv = sv_all + off;
            const double* sval = sval_all + off;
            const int* lc = lc_all + off;
            const int* rc = rc_all + off;
            const double* lm = lm_all + off;
            int leaf_off = bb * n_hon;

            bool tree_has_split = (sp_all[var_col * B + bb] != 0);

            if (tree_has_split) {
                // Shared-prefix routing
                int node = 0;
                while (lc[node] != 0 || rc[node] != 0) {
                    if (sv[node] == var_col) break;
                    node = (X_ptr[xbase + sv[node]] <= sval[node]) ? lc[node] : rc[node];
                }

                int leaf_a_node, leaf_b_node;
                if (lc[node] == 0 && rc[node] == 0) {
                    int lo = ol_all[leaf_off + j];
                    leaf_a_node = lo; leaf_b_node = lo;
                } else {
                    int node_a = (a <= sval[node]) ? lc[node] : rc[node];
                    int node_b = (b <= sval[node]) ? lc[node] : rc[node];
                    while (lc[node_a] != 0 || rc[node_a] != 0) {
                        double xv = (sv[node_a] == var_col) ? a : X_ptr[xbase + sv[node_a]];
                        node_a = (xv <= sval[node_a]) ? lc[node_a] : rc[node_a];
                    }
                    leaf_a_node = node_a;
                    while (lc[node_b] != 0 || rc[node_b] != 0) {
                        double xv = (sv[node_b] == var_col) ? b : X_ptr[xbase + sv[node_b]];
                        node_b = (xv <= sval[node_b]) ? lc[node_b] : rc[node_b];
                    }
                    leaf_b_node = node_b;
                }
                if (lm[leaf_a_node] != LEAF_EMPTY) { fa_sum += lm[leaf_a_node]; fa_cnt++; }
                if (lm[leaf_b_node] != LEAF_EMPTY) { fb_sum += lm[leaf_b_node]; fb_cnt++; }
                if (leaf_a_node != leaf_b_node) n_diff_leaves++;
            } else {
                int leaf = ol_all[leaf_off + j];
                if (lm[leaf] != LEAF_EMPTY) {
                    double v = lm[leaf];
                    fa_sum += v; fa_cnt++;
                    fb_sum += v; fb_cnt++;
                }
            }
        }

        double fa = (fa_cnt > 0) ? fa_sum / fa_cnt : NA_REAL;
        double fb = (fb_cnt > 0) ? fb_sum / fb_cnt : NA_REAL;
        double fo = fhat_obs_cached[j];
        fhat_a_out[j] = fa; fhat_b_out[j] = fb; fhat_obs_out[j] = fo;

        if (ISNA(fa) || ISNA(fb) || ISNA(fo)) { phi[j] = NA_REAL; }
        else {
            double pc = fa - fb;
            double res = y_ptr[i] - fo;
            double xj = has_indicator ? ind_ptr[i] : X_ptr[j * p + var_col];
            double gi = g_ptr[i], co;
            if (is_binary) {
                double gc = std::max(0.025, std::min(0.975, gi));
                double w = (xj - gc) / (gc * (1.0 - gc));
                co = w * res;
            } else {
                double ej = xj - gi;
                double w = ej / sigma2_ej;
                co = w * res * (a - b);
            }
            phi[j] = pc + co;
            sum_pc += pc; sum_co += co;
            psi_sum += phi[j]; psi_cnt++;
        }
    }

    double psi = (psi_cnt > 0) ? psi_sum / psi_cnt : NA_REAL;
    return List::create(
        Named("psi") = psi, Named("phi") = phi,
        Named("mean_pred_contrast") = (psi_cnt > 0) ? sum_pc / psi_cnt : NA_REAL,
        Named("mean_correction") = (psi_cnt > 0) ? sum_co / psi_cnt : NA_REAL,
        Named("n_contributing") = psi_cnt,
        Named("n_split_trees") = n_split_trees,
        Named("n_diff_leaves") = n_diff_leaves,
        Named("n_trees") = B,
        Named("fhat_a") = fhat_a_out, Named("fhat_b") = fhat_b_out,
        Named("fhat_obs") = fhat_obs_out, Named("sigma2_ej") = sigma2_ej);
}


// ============================================================
// Phase 3: Multi-variable AIPW scorer
// One factual walk per obs-tree finds all divergence nodes.
// Counterfactual sub-walks only for variables on the path.
// fhat_obs precomputed in cache — not recomputed here.
// ============================================================

// [[Rcpp::export]]
List aipw_scores_multi_cpp(
    List cache,
    IntegerVector var_cols,      // 0-based column indices for all variables
    LogicalVector is_binary_vec, // is each variable binary?
    NumericVector a_vals,        // counterfactual 'a' per variable
    NumericVector b_vals,        // counterfactual 'b' per variable
    List ghat_list,              // propensity vector per variable
    Nullable<List> indicator_list_ = R_NilValue  // optional indicator per variable (categorical)
) {
    int B = as<int>(cache["B"]);
    int n = as<int>(cache["n"]);
    int p = as<int>(cache["p"]);
    int n_hon = as<int>(cache["n_hon"]);
    IntegerVector honest_idx = as<IntegerVector>(cache["honest_idx"]);
    NumericVector y_honest = as<NumericVector>(cache["y_honest"]);
    IntegerVector tree_offsets = as<IntegerVector>(cache["tree_offsets"]);
    IntegerVector flat_sv = as<IntegerVector>(cache["flat_sv"]);
    NumericVector flat_sval = as<NumericVector>(cache["flat_sval"]);
    IntegerVector flat_lc = as<IntegerVector>(cache["flat_lc"]);
    IntegerVector flat_rc = as<IntegerVector>(cache["flat_rc"]);
    NumericVector X_row_flat = as<NumericVector>(cache["X_row_flat"]);
    const double* X_ptr = REAL(X_row_flat);
    IntegerVector obs_leaf_flat = as<IntegerVector>(cache["obs_leaf_flat"]);
    NumericVector leaf_mean_flat = as<NumericVector>(cache["leaf_mean_flat"]);
    NumericVector fhat_obs_cached = as<NumericVector>(cache["fhat_obs"]);
    const double* y_ptr = REAL(y_honest);

    int n_vars = var_cols.size();
    const int* vc = INTEGER(var_cols);
    const double* a_ptr = REAL(a_vals);
    const double* b_ptr = REAL(b_vals);

    // Propensity pointers per variable
    std::vector<const double*> g_ptrs(n_vars);
    for (int v = 0; v < n_vars; v++) {
        NumericVector gv = as<NumericVector>(ghat_list[v]);
        g_ptrs[v] = REAL(gv);
    }

    // Optional indicator pointers per variable (for categorical)
    bool has_indicators = indicator_list_.isNotNull();
    List indicator_list;
    std::vector<const double*> ind_ptrs(n_vars, nullptr);
    std::vector<bool> has_ind(n_vars, false);
    if (has_indicators) {
        indicator_list = as<List>(indicator_list_);
        for (int v = 0; v < n_vars; v++) {
            if (!Rf_isNull(indicator_list[v])) {
                NumericVector iv = as<NumericVector>(indicator_list[v]);
                ind_ptrs[v] = REAL(iv);
                has_ind[v] = true;
            }
        }
    }

    // Quick lookup: var_col -> variable index (-1 if not in set)
    std::vector<int> col_to_var(p, -1);
    for (int v = 0; v < n_vars; v++) {
        col_to_var[vc[v]] = v;
    }

    // Per-variable, per-observation accumulators for counterfactual predictions
    // fhat_a[v][i], fhat_b[v][i] — indexed by original obs index
    std::vector<std::vector<double>> fhat_a_sum(n_vars, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> fhat_b_sum(n_vars, std::vector<double>(n, 0.0));
    std::vector<std::vector<int>> fhat_a_cnt(n_vars, std::vector<int>(n, 0));
    std::vector<std::vector<int>> fhat_b_cnt(n_vars, std::vector<int>(n, 0));

    // Temporary: divergence tracking
    struct DivNode { int var_idx; int node; int child_taken; };
    std::vector<DivNode> div_nodes(64);  // generous preallocation
    std::vector<bool> var_on_path(n_vars, false);  // reused per obs-tree

    for (int bb = 0; bb < B; bb++) {
        int off = tree_offsets[bb];
        const int* sv = INTEGER(flat_sv) + off;
        const double* sval = REAL(flat_sval) + off;
        const int* lc = INTEGER(flat_lc) + off;
        const int* rc = INTEGER(flat_rc) + off;
        const double* lm = REAL(leaf_mean_flat) + off;

        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            int xbase = j * p;

            // --- One factual walk: root to leaf ---
            int n_div = 0;
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0) {
                int col = sv[node];
                double xv = X_ptr[xbase + col];
                bool go_left = (xv <= sval[node]);
                int child = go_left ? lc[node] : rc[node];

                if (col < p && col_to_var[col] >= 0) {
                    if (n_div < (int)div_nodes.size()) {
                        div_nodes[n_div] = {col_to_var[col], node, go_left ? 0 : 1};
                    }
                    n_div++;
                }
                node = child;
            }
            if (n_div > (int)div_nodes.size()) n_div = (int)div_nodes.size();

            double obs_lm = lm[node];
            if (obs_lm != LEAF_EMPTY) {
                for (int v = 0; v < n_vars; v++) {
                    fhat_a_sum[v][i] += obs_lm;
                    fhat_a_cnt[v][i]++;
                    fhat_b_sum[v][i] += obs_lm;
                    fhat_b_cnt[v][i]++;
                }
            }

            // Mark which variables are on this path
            for (int v = 0; v < n_vars; v++) var_on_path[v] = false;
            for (int d = 0; d < n_div; d++) var_on_path[div_nodes[d].var_idx] = true;

            for (int v = 0; v < n_vars; v++) {
                if (!var_on_path[v]) continue;  // no split on this var → leaf_a = leaf_b = obs_leaf

                int vc_v = vc[v];
                double av = a_ptr[v], bv = b_ptr[v];

                // Full counterfactual walk for 'a'
                int node_a = 0;
                while (lc[node_a] != 0 || rc[node_a] != 0) {
                    double xv = (sv[node_a] == vc_v) ? av : X_ptr[xbase + sv[node_a]];
                    node_a = (xv <= sval[node_a]) ? lc[node_a] : rc[node_a];
                }
                // Correct: subtract obs_leaf, add counterfactual leaf
                if (obs_lm != LEAF_EMPTY) { fhat_a_sum[v][i] -= obs_lm; fhat_a_cnt[v][i]--; }
                if (lm[node_a] != LEAF_EMPTY) { fhat_a_sum[v][i] += lm[node_a]; fhat_a_cnt[v][i]++; }

                // Full counterfactual walk for 'b'
                int node_b = 0;
                while (lc[node_b] != 0 || rc[node_b] != 0) {
                    double xv = (sv[node_b] == vc_v) ? bv : X_ptr[xbase + sv[node_b]];
                    node_b = (xv <= sval[node_b]) ? lc[node_b] : rc[node_b];
                }
                if (obs_lm != LEAF_EMPTY) { fhat_b_sum[v][i] -= obs_lm; fhat_b_cnt[v][i]--; }
                if (lm[node_b] != LEAF_EMPTY) { fhat_b_sum[v][i] += lm[node_b]; fhat_b_cnt[v][i]++; }
            }
        }
    }

    // --- Compute phi scores per variable ---
    List results(n_vars);
    for (int v = 0; v < n_vars; v++) {
        NumericVector phi(n_hon);
        double psi_sum = 0.0; int psi_cnt = 0;
        double sum_pc = 0.0, sum_co = 0.0;

        bool is_bin = is_binary_vec[v];
        double av = a_ptr[v], bv = b_ptr[v];
        const double* g_ptr_v = g_ptrs[v];

        double sigma2_ej = 0.0;
        if (!is_bin) {
            double ss = 0.0;
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1;
                double xj_val = has_ind[v] ? ind_ptrs[v][i] : X_ptr[j * p + vc[v]];
                double ej = xj_val - g_ptr_v[i];
                ss += ej * ej;
            }
            sigma2_ej = ss / n_hon;
        }

        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double fa = (fhat_a_cnt[v][i] > 0) ? fhat_a_sum[v][i] / fhat_a_cnt[v][i] : NA_REAL;
            double fb = (fhat_b_cnt[v][i] > 0) ? fhat_b_sum[v][i] / fhat_b_cnt[v][i] : NA_REAL;
            double fo = fhat_obs_cached[j];  // precomputed
            if (ISNA(fa) || ISNA(fb) || ISNA(fo)) { phi[j] = NA_REAL; continue; }

            double pc = fa - fb;
            double res = y_ptr[i] - fo;
            double xj = has_ind[v] ? ind_ptrs[v][i] : X_ptr[j * p + vc[v]];
            double gi = g_ptr_v[i], w, co;
            if (is_bin) {
                double gc = std::max(0.025, std::min(0.975, gi));
                w = (xj - gc) / (gc * (1.0 - gc));
                co = w * res;
            } else {
                double ej = xj - gi;
                w = ej / sigma2_ej;
                co = w * res * (av - bv);
            }
            phi[j] = pc + co;
            sum_pc += pc; sum_co += co;
            psi_sum += phi[j]; psi_cnt++;
        }

        double psi = (psi_cnt > 0) ? psi_sum / psi_cnt : NA_REAL;
        results[v] = List::create(
            Named("psi") = psi, Named("phi") = phi,
            Named("mean_pred_contrast") = (psi_cnt > 0) ? sum_pc / psi_cnt : NA_REAL,
            Named("mean_correction") = (psi_cnt > 0) ? sum_co / psi_cnt : NA_REAL,
            Named("n_contributing") = psi_cnt);
    }

    return results;
}


// ============================================================
// Phase 4: Cached curve scorer for continuous variables
// Uses precomputed cache: flat tree arrays, obs_leaf, leaf means,
// fhat_obs, row-major X. Only does grid-point counterfactual routing.
// ============================================================

// [[Rcpp::export]]
List aipw_curve_cached_cpp(
    List cache,
    NumericVector ghat,
    int var_col,
    NumericVector grid_points,
    double sigma2_override = -1.0,
    int n_threads = 1
) {
    (void)n_threads;
    int B = as<int>(cache["B"]);
    int n = as<int>(cache["n"]); (void)n;
    int p = as<int>(cache["p"]);
    int n_hon = as<int>(cache["n_hon"]);
    IntegerVector honest_idx = as<IntegerVector>(cache["honest_idx"]);
    NumericVector y_honest = as<NumericVector>(cache["y_honest"]);
    IntegerVector tree_offsets = as<IntegerVector>(cache["tree_offsets"]);
    IntegerVector flat_sv = as<IntegerVector>(cache["flat_sv"]);
    NumericVector flat_sval = as<NumericVector>(cache["flat_sval"]);
    IntegerVector flat_lc = as<IntegerVector>(cache["flat_lc"]);
    IntegerVector flat_rc = as<IntegerVector>(cache["flat_rc"]);
    NumericVector X_row_flat = as<NumericVector>(cache["X_row_flat"]);
    const double* X_ptr = REAL(X_row_flat);
    IntegerVector obs_leaf_flat = as<IntegerVector>(cache["obs_leaf_flat"]);
    NumericVector leaf_mean_flat = as<NumericVector>(cache["leaf_mean_flat"]);
    IntegerVector splits_on = as<IntegerVector>(cache["splits_on"]);
    NumericVector fhat_obs_cached = as<NumericVector>(cache["fhat_obs"]);
    const double* y_ptr = REAL(y_honest);
    const double* g_ptr = REAL(ghat);
    const double* grid_ptr = REAL(grid_points);
    const int* off_arr = INTEGER(tree_offsets);
    const int* sv_all = INTEGER(flat_sv);
    const double* sval_all = REAL(flat_sval);
    const int* lc_all = INTEGER(flat_lc);
    const int* rc_all = INTEGER(flat_rc);
    const double* lm_all = REAL(leaf_mean_flat);
    const int* ol_all = INTEGER(obs_leaf_flat);
    const int* sp_all = INTEGER(splits_on);

    int G_plus_1 = grid_points.size();
    int G = G_plus_1 - 1;

    int n_split_trees = 0;
    for (int bb = 0; bb < B; bb++)
        if (sp_all[var_col * B + bb] != 0) n_split_trees++;

    // Compute sigma2_ej
    double sigma2_ej;
    if (sigma2_override > 0.0) {
        sigma2_ej = sigma2_override;
    } else {
        double ss = 0.0;
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double ej = X_ptr[j * p + var_col] - g_ptr[i];
            ss += ej * ej;
        }
        sigma2_ej = ss / n_hon;
    }

    // Per-obs, per-grid-point arrays: fhat_grid(j, g) and fhat_obs(j)
    // Compute in parallel over observations
    NumericMatrix fhat_grid(n_hon, G_plus_1);
    NumericVector fhat_obs(n_hon);
    NumericMatrix phi_scores(n_hon, G);
    std::fill(phi_scores.begin(), phi_scores.end(), NA_REAL);


    // Parallel over observations: accumulate grid predictions across all trees
    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;
        int xbase = j * p;
        fhat_obs[j] = fhat_obs_cached[j];

        // Per-grid accumulators for this observation
        std::vector<double> gsum(G_plus_1, 0.0);
        std::vector<int> gcnt(G_plus_1, 0);

        for (int bb = 0; bb < B; bb++) {
            int off = off_arr[bb];
            const int* sv = sv_all + off;
            const double* sval = sval_all + off;
            const int* lc = lc_all + off;
            const int* rc = rc_all + off;
            const double* lm = lm_all + off;
            int leaf_off = bb * n_hon;

            bool tree_has_split = (sp_all[var_col * B + bb] != 0);

            if (tree_has_split) {
                // Shared-prefix: walk to first var_col split
                int prefix_node = 0;
                while (lc[prefix_node] != 0 || rc[prefix_node] != 0) {
                    if (sv[prefix_node] == var_col) break;
                    prefix_node = (X_ptr[xbase + sv[prefix_node]] <= sval[prefix_node]) ?
                                  lc[prefix_node] : rc[prefix_node];
                }

                if (lc[prefix_node] == 0 && rc[prefix_node] == 0) {
                    if (lm[prefix_node] != LEAF_EMPTY) {
                        double v = lm[prefix_node];
                        for (int g = 0; g < G_plus_1; g++) { gsum[g] += v; gcnt[g]++; }
                    }
                } else {
                    for (int g = 0; g < G_plus_1; g++) {
                        double gval = grid_ptr[g];
                        int node = (gval <= sval[prefix_node]) ?
                                   lc[prefix_node] : rc[prefix_node];
                        while (lc[node] != 0 || rc[node] != 0) {
                            double xv = (sv[node] == var_col) ? gval : X_ptr[xbase + sv[node]];
                            node = (xv <= sval[node]) ? lc[node] : rc[node];
                        }
                        if (lm[node] != LEAF_EMPTY) { gsum[g] += lm[node]; gcnt[g]++; }
                    }
                }
            } else {
                int leaf = ol_all[leaf_off + j];
                if (lm[leaf] != LEAF_EMPTY) {
                    double v = lm[leaf];
                    for (int g = 0; g < G_plus_1; g++) { gsum[g] += v; gcnt[g]++; }
                }
            }
        }

        // Store grid predictions for this observation
        for (int g = 0; g < G_plus_1; g++)
            fhat_grid(j, g) = (gcnt[g] > 0) ? gsum[g] / gcnt[g] : NA_REAL;

        // Compute phi scores for each interval
        double fo = fhat_obs_cached[j];
        double res = y_ptr[i] - fo;
        double ej = X_ptr[j * p + var_col] - g_ptr[i];
        double w = ej / sigma2_ej;

        for (int g = 0; g < G; g++) {
            double fh = fhat_grid(j, g + 1);
            double fl = fhat_grid(j, g);
            if (ISNA(fh) || ISNA(fl) || ISNA(fo)) continue;
            double dg = grid_ptr[g + 1] - grid_ptr[g];
            double pc = fh - fl;
            double pg = pc + w * res * dg;
            phi_scores(j, g) = pg / dg;
        }
    }

    // Aggregate slopes across observations (sequential — small loop)
    NumericVector slopes(G), pred_slopes(G);
    for (int g = 0; g < G; g++) {
        double dg = grid_ptr[g + 1] - grid_ptr[g];
        double phi_sum = 0.0, pred_sum_g = 0.0; int cnt = 0;
        for (int j = 0; j < n_hon; j++) {
            if (ISNA(phi_scores(j, g))) continue;
            double fh = fhat_grid(j, g + 1), fl = fhat_grid(j, g);
            if (ISNA(fh) || ISNA(fl)) continue;
            phi_sum += phi_scores(j, g) * dg;
            pred_sum_g += (fh - fl);
            cnt++;
        }
        slopes[g] = (cnt > 0) ? (phi_sum / cnt) / dg : NA_REAL;
        pred_slopes[g] = (cnt > 0) ? (pred_sum_g / cnt) / dg : NA_REAL;
    }

    NumericVector curve(G_plus_1, 0.0);
    for (int g = 0; g < G; g++) {
        double dg = grid_ptr[g + 1] - grid_ptr[g];
        curve[g + 1] = curve[g] + (ISNA(slopes[g]) ? 0.0 : slopes[g] * dg);
    }

    return List::create(
        Named("slopes") = slopes, Named("pred_slopes") = pred_slopes,
        Named("curve") = curve, Named("grid") = grid_points,
        Named("sigma2_ej") = sigma2_ej, Named("n_hon") = n_hon,
        Named("n_split_trees") = n_split_trees, Named("n_trees") = B,
        Named("phi_scores") = phi_scores,
        Named("fhat_grid") = fhat_grid, Named("fhat_obs") = fhat_obs);
}

// ============================================================
// honest_predict_cached_cpp
//
// Routes query observations through cached flat tree arrays and
// returns honest predictions using precomputed leaf means.
// Eliminates the need to store ranger forest objects for PASR.
//
// X_query: n_query x p matrix (column-major, ranger column order)
// cache: precomputed forest cache from precompute_forest_cache_cpp
//
// Returns: NumericVector of length n_query with honest predictions
// ============================================================

// [[Rcpp::export]]
NumericVector honest_predict_cached_cpp(
    List cache, NumericMatrix X_query, int n_threads = 1
) {
    (void)n_threads;
    int B = as<int>(cache["B"]);
    int p = as<int>(cache["p"]); (void)p;
    IntegerVector tree_offsets = as<IntegerVector>(cache["tree_offsets"]);
    IntegerVector flat_sv = as<IntegerVector>(cache["flat_sv"]);
    NumericVector flat_sval = as<NumericVector>(cache["flat_sval"]);
    IntegerVector flat_lc = as<IntegerVector>(cache["flat_lc"]);
    IntegerVector flat_rc = as<IntegerVector>(cache["flat_rc"]);
    NumericVector leaf_mean_flat = as<NumericVector>(cache["leaf_mean_flat"]);

    int n_query = X_query.nrow();
    const double* Xq_ptr = REAL(X_query);
    const int* sv_ptr = INTEGER(flat_sv);
    const double* sval_ptr = REAL(flat_sval);
    const int* lc_ptr = INTEGER(flat_lc);
    const int* rc_ptr = INTEGER(flat_rc);
    const double* lm_ptr = REAL(leaf_mean_flat);
    const int* off_ptr = INTEGER(tree_offsets);

    // Parallelize over query points — each query is independent
    std::vector<double> result_vec(n_query);


    for (int q = 0; q < n_query; q++) {
        double psum = 0.0;
        int pcnt = 0;
        for (int b = 0; b < B; b++) {
            int off = off_ptr[b];
            int node = 0;
            while (lc_ptr[off + node] != 0 || rc_ptr[off + node] != 0) {
                int var = sv_ptr[off + node];
                double val = Xq_ptr[q + n_query * var];
                node = (val <= sval_ptr[off + node]) ? lc_ptr[off + node] : rc_ptr[off + node];
            }
            double lm = lm_ptr[off + node];
            if (lm != -1e308) { psum += lm; pcnt++; }
        }
        result_vec[q] = (pcnt > 0) ? psum / pcnt : NA_REAL;
    }

    NumericVector result(n_query);
    for (int q = 0; q < n_query; q++) result[q] = result_vec[q];
    return result;
}


// ============================================================
// pasr_extract_all_binary_cpp
//
// Batch extraction: given R replicates × 4 caches each,
// extracts psi_A and psi_B for binary effect in one C++ call.
// Eliminates 400 R-to-C++ call overhead.
//
// cache_list: List of R elements, each a list of 4 caches
//             (A_AB, A_BA, B_AB, B_BA)
// ghat: propensity vector (length n)
// var_col: 0-based column index
// n_threads: OpenMP threads
//
// Returns: List with psi_A (length R), psi_B (length R)
// ============================================================

// [[Rcpp::export]]
List pasr_extract_all_binary_cpp(
    List cache_list, NumericVector ghat, int var_col,
    double code_to = 1.0, double code_from = 0.0,
    Nullable<NumericVector> indicator_ = R_NilValue,
    int n_threads = 1
) {
    int R = cache_list.size();
    const double* g_ptr = REAL(ghat);
    bool has_indicator = indicator_.isNotNull();
    const double* ind_ptr = has_indicator ? REAL(as<NumericVector>(indicator_)) : nullptr;

    // Pre-extract all cache data into raw C++ arrays BEFORE parallel region
    struct CacheData {
        int B, n, p, n_hon;
        const int* honest_idx;
        const double* y_honest;
        const int* tree_offsets;
        const int* flat_sv;
        const double* flat_sval;
        const int* flat_lc;
        const int* flat_rc;
        const double* X_row_flat;
        const int* obs_leaf_flat;
        const double* leaf_mean_flat;
        const int* splits_on;
        const double* fhat_obs;
    };

    // 4 directions per replicate: A_AB, A_BA, B_AB, B_BA
    std::vector<std::vector<CacheData>> all_caches(R, std::vector<CacheData>(4));
    // Keep Rcpp objects alive
    std::vector<std::vector<List>> cache_holders(R, std::vector<List>(4));

    const char* dir_names[4] = {"A_AB", "A_BA", "B_AB", "B_BA"};
    for (int r = 0; r < R; r++) {
        List rep = cache_list[r];
        for (int d = 0; d < 4; d++) {
            List c = rep[dir_names[d]];
            cache_holders[r][d] = c;
            CacheData& cd = all_caches[r][d];
            cd.B = as<int>(c["B"]);
            cd.n = as<int>(c["n"]);
            cd.p = as<int>(c["p"]);
            cd.n_hon = as<int>(c["n_hon"]);
            cd.honest_idx = INTEGER(as<IntegerVector>(c["honest_idx"]));
            cd.y_honest = REAL(as<NumericVector>(c["y_honest"]));
            cd.tree_offsets = INTEGER(as<IntegerVector>(c["tree_offsets"]));
            cd.flat_sv = INTEGER(as<IntegerVector>(c["flat_sv"]));
            cd.flat_sval = REAL(as<NumericVector>(c["flat_sval"]));
            cd.flat_lc = INTEGER(as<IntegerVector>(c["flat_lc"]));
            cd.flat_rc = INTEGER(as<IntegerVector>(c["flat_rc"]));
            cd.X_row_flat = REAL(as<NumericVector>(c["X_row_flat"]));
            cd.obs_leaf_flat = INTEGER(as<IntegerVector>(c["obs_leaf_flat"]));
            cd.leaf_mean_flat = REAL(as<NumericVector>(c["leaf_mean_flat"]));
            cd.splits_on = INTEGER(as<IntegerVector>(c["splits_on"]));
            cd.fhat_obs = REAL(as<NumericVector>(c["fhat_obs"]));
        }
    }

    std::vector<double> psi_A_vec(R), psi_B_vec(R);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int r = 0; r < R; r++) {
        double dir_psi[4];

        for (int d = 0; d < 4; d++) {
            const CacheData& cd = all_caches[r][d];
            int B = cd.B, n = cd.n, p = cd.p, n_hon = cd.n_hon;

            std::vector<double> fa_sum(n, 0.0), fb_sum(n, 0.0);
            std::vector<int> fa_cnt(n, 0), fb_cnt(n, 0);

            for (int bb = 0; bb < B; bb++) {
                int off = cd.tree_offsets[bb];
                const int* sv = cd.flat_sv + off;
                const double* sval = cd.flat_sval + off;
                const int* lc = cd.flat_lc + off;
                const int* rc = cd.flat_rc + off;
                const double* lm = cd.leaf_mean_flat + off;
                int leaf_off = bb * n_hon;
                bool tree_has_split = (cd.splits_on[var_col * B + bb] != 0);

                if (tree_has_split) {
                    for (int j = 0; j < n_hon; j++) {
                        int i = cd.honest_idx[j] - 1;
                        int xbase = j * p;
                        int node = 0;
                        while (lc[node] != 0 || rc[node] != 0) {
                            if (sv[node] == var_col) break;
                            node = (cd.X_row_flat[xbase + sv[node]] <= sval[node]) ? lc[node] : rc[node];
                        }
                        int la, lb;
                        if (lc[node] == 0 && rc[node] == 0) {
                            la = cd.obs_leaf_flat[leaf_off + j]; lb = la;
                        } else {
                            int na = (code_to <= sval[node]) ? lc[node] : rc[node];
                            int nb = (code_from <= sval[node]) ? lc[node] : rc[node];
                            while (lc[na] != 0 || rc[na] != 0) {
                                double xv = (sv[na] == var_col) ? code_to : cd.X_row_flat[xbase + sv[na]];
                                na = (xv <= sval[na]) ? lc[na] : rc[na];
                            }
                            la = na;
                            while (lc[nb] != 0 || rc[nb] != 0) {
                                double xv = (sv[nb] == var_col) ? code_from : cd.X_row_flat[xbase + sv[nb]];
                                nb = (xv <= sval[nb]) ? lc[nb] : rc[nb];
                            }
                            lb = nb;
                        }
                        if (lm[la] != -1e308) { fa_sum[i] += lm[la]; fa_cnt[i]++; }
                        if (lm[lb] != -1e308) { fb_sum[i] += lm[lb]; fb_cnt[i]++; }
                    }
                } else {
                    for (int j = 0; j < n_hon; j++) {
                        int i = cd.honest_idx[j] - 1;
                        int leaf = cd.obs_leaf_flat[leaf_off + j];
                        if (lm[leaf] != -1e308) {
                            double v = lm[leaf];
                            fa_sum[i] += v; fa_cnt[i]++;
                            fb_sum[i] += v; fb_cnt[i]++;
                        }
                    }
                }
            }

            double psi_sum = 0.0; int psi_cnt = 0;
            for (int j = 0; j < n_hon; j++) {
                int i = cd.honest_idx[j] - 1;
                double fa = (fa_cnt[i] > 0) ? fa_sum[i] / fa_cnt[i] : NA_REAL;
                double fb = (fb_cnt[i] > 0) ? fb_sum[i] / fb_cnt[i] : NA_REAL;
                double fo = cd.fhat_obs[j];
                if (ISNA(fa) || ISNA(fb) || ISNA(fo)) continue;
                double pc = fa - fb;
                double res = cd.y_honest[i] - fo;
                double xj = has_indicator ? ind_ptr[i] : cd.X_row_flat[j * p + var_col];
                if (ISNA(xj)) continue;  // skip obs not in pair
                double gc = g_ptr[i];
                if (ISNA(gc)) continue;  // skip obs without propensity
                gc = std::max(0.025, std::min(0.975, gc));
                double w = (xj - gc) / (gc * (1.0 - gc));
                psi_sum += pc + w * res; psi_cnt++;
            }
            dir_psi[d] = (psi_cnt > 0) ? psi_sum / psi_cnt : NA_REAL;
        }

        psi_A_vec[r] = (dir_psi[0] + dir_psi[1]) / 2.0;
        psi_B_vec[r] = (dir_psi[2] + dir_psi[3]) / 2.0;
    }

    NumericVector psi_A(R), psi_B(R);
    for (int r = 0; r < R; r++) { psi_A[r] = psi_A_vec[r]; psi_B[r] = psi_B_vec[r]; }
    return List::create(Named("psi_A") = psi_A, Named("psi_B") = psi_B);
}


// ============================================================
// pasr_extract_all_continuous_cpp
//
// Batch extraction for continuous effects: given R caches,
// extracts psi_A and psi_B via curve integration.
// ============================================================

// [[Rcpp::export]]
List pasr_extract_all_continuous_cpp(
    List cache_list, NumericVector ghat, int var_col,
    NumericVector grid_points, double a_val, double b_val,
    int n_threads = 1
) {
    int R = cache_list.size();
    int G_plus_1 = grid_points.size();
    int G = G_plus_1 - 1;
    const double* grid_ptr = REAL(grid_points);
    const double* g_ptr = REAL(ghat);

    // Reuse same CacheData struct
    struct CacheDataC {
        int B, n, p, n_hon;
        const int* honest_idx;
        const double* y_honest;
        const int* tree_offsets;
        const int* flat_sv;
        const double* flat_sval;
        const int* flat_lc;
        const int* flat_rc;
        const double* X_row_flat;
        const int* obs_leaf_flat;
        const double* leaf_mean_flat;
        const int* splits_on;
        const double* fhat_obs;
    };

    std::vector<std::vector<CacheDataC>> all_caches(R, std::vector<CacheDataC>(4));
    std::vector<std::vector<List>> cache_holders(R, std::vector<List>(4));

    const char* dir_names[4] = {"A_AB", "A_BA", "B_AB", "B_BA"};
    for (int r = 0; r < R; r++) {
        List rep = cache_list[r];
        for (int d = 0; d < 4; d++) {
            List c = rep[dir_names[d]];
            cache_holders[r][d] = c;
            CacheDataC& cd = all_caches[r][d];
            cd.B = as<int>(c["B"]);
            cd.n = as<int>(c["n"]);
            cd.p = as<int>(c["p"]);
            cd.n_hon = as<int>(c["n_hon"]);
            cd.honest_idx = INTEGER(as<IntegerVector>(c["honest_idx"]));
            cd.y_honest = REAL(as<NumericVector>(c["y_honest"]));
            cd.tree_offsets = INTEGER(as<IntegerVector>(c["tree_offsets"]));
            cd.flat_sv = INTEGER(as<IntegerVector>(c["flat_sv"]));
            cd.flat_sval = REAL(as<NumericVector>(c["flat_sval"]));
            cd.flat_lc = INTEGER(as<IntegerVector>(c["flat_lc"]));
            cd.flat_rc = INTEGER(as<IntegerVector>(c["flat_rc"]));
            cd.X_row_flat = REAL(as<NumericVector>(c["X_row_flat"]));
            cd.obs_leaf_flat = INTEGER(as<IntegerVector>(c["obs_leaf_flat"]));
            cd.leaf_mean_flat = REAL(as<NumericVector>(c["leaf_mean_flat"]));
            cd.splits_on = INTEGER(as<IntegerVector>(c["splits_on"]));
            cd.fhat_obs = REAL(as<NumericVector>(c["fhat_obs"]));
        }
    }

    std::vector<double> psi_A_vec(R), psi_B_vec(R);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int r = 0; r < R; r++) {
        double dir_psi[4];

        for (int d = 0; d < 4; d++) {
            const CacheDataC& cd = all_caches[r][d];
            int B = cd.B, n = cd.n, p = cd.p, n_hon = cd.n_hon;

            double ss = 0.0;
            for (int j = 0; j < n_hon; j++) {
                int i = cd.honest_idx[j] - 1;
                double ej = cd.X_row_flat[j * p + var_col] - g_ptr[i];
                ss += ej * ej;
            }
            double sigma2_ej = ss / n_hon;

            // Flat allocation: gsum[g * n + i], gcnt[g * n + i]
            std::vector<double> gsum(G_plus_1 * n, 0.0);
            std::vector<int> gcnt(G_plus_1 * n, 0);

            for (int bb = 0; bb < B; bb++) {
                int off = cd.tree_offsets[bb];
                const int* sv = cd.flat_sv + off;
                const double* sval = cd.flat_sval + off;
                const int* lc = cd.flat_lc + off;
                const int* rc = cd.flat_rc + off;
                const double* lm = cd.leaf_mean_flat + off;
                int leaf_off = bb * n_hon;
                bool tree_has_split = (cd.splits_on[var_col * B + bb] != 0);

                if (tree_has_split) {
                    for (int j = 0; j < n_hon; j++) {
                        int i = cd.honest_idx[j] - 1;
                        int xbase = j * p;

                        // Shared-prefix: walk to first var_col split
                        int prefix_node = 0;
                        while (lc[prefix_node] != 0 || rc[prefix_node] != 0) {
                            if (sv[prefix_node] == var_col) break;
                            prefix_node = (cd.X_row_flat[xbase + sv[prefix_node]] <= sval[prefix_node]) ?
                                          lc[prefix_node] : rc[prefix_node];
                        }

                        if (lc[prefix_node] == 0 && rc[prefix_node] == 0) {
                            // No var_col split on this path — same leaf for all grid
                            if (lm[prefix_node] != -1e308) {
                                double v = lm[prefix_node];
                                for (int g = 0; g < G_plus_1; g++) {
                                    gsum[g * n + i] += v; gcnt[g * n + i]++;
                                }
                            }
                        } else {
                            // Branch from prefix_node per grid point
                            for (int g = 0; g < G_plus_1; g++) {
                                double gval = grid_ptr[g];
                                int node = (gval <= sval[prefix_node]) ?
                                           lc[prefix_node] : rc[prefix_node];
                                while (lc[node] != 0 || rc[node] != 0) {
                                    double xv = (sv[node] == var_col) ? gval :
                                                cd.X_row_flat[xbase + sv[node]];
                                    node = (xv <= sval[node]) ? lc[node] : rc[node];
                                }
                                if (lm[node] != -1e308) {
                                    gsum[g * n + i] += lm[node]; gcnt[g * n + i]++;
                                }
                            }
                        }
                    }
                } else {
                    for (int j = 0; j < n_hon; j++) {
                        int i = cd.honest_idx[j] - 1;
                        int leaf = cd.obs_leaf_flat[leaf_off + j];
                        if (lm[leaf] != -1e308) {
                            double v = lm[leaf];
                            for (int g = 0; g < G_plus_1; g++) { gsum[g * n + i] += v; gcnt[g * n + i]++; }
                        }
                    }
                }
            }

            std::vector<double> slopes(G, 0.0);
            for (int g = 0; g < G; g++) {
                double dg = grid_ptr[g + 1] - grid_ptr[g];
                double phi_sum = 0.0; int cnt = 0;
                for (int j = 0; j < n_hon; j++) {
                    int i = cd.honest_idx[j] - 1;
                    double fh = (gcnt[(g+1)*n+i] > 0) ? gsum[(g+1)*n+i] / gcnt[(g+1)*n+i] : NA_REAL;
                    double fl = (gcnt[g*n+i] > 0) ? gsum[g*n+i] / gcnt[g*n+i] : NA_REAL;
                    double fo = cd.fhat_obs[j];
                    if (ISNA(fh) || ISNA(fl) || ISNA(fo)) continue;
                    double pc = fh - fl;
                    double res = cd.y_honest[i] - fo;
                    double ej = cd.X_row_flat[j * p + var_col] - g_ptr[i];
                    phi_sum += pc + (ej / sigma2_ej) * res * dg;
                    cnt++;
                }
                slopes[g] = (cnt > 0) ? (phi_sum / cnt) / dg : 0.0;
            }

            std::vector<double> curve(G_plus_1, 0.0);
            for (int g = 0; g < G; g++)
                curve[g+1] = curve[g] + slopes[g] * (grid_ptr[g+1] - grid_ptr[g]);

            double va = curve[G];
            for (int g = 0; g < G; g++) {
                if (a_val <= grid_ptr[g+1]) {
                    double frac = (a_val - grid_ptr[g]) / (grid_ptr[g+1] - grid_ptr[g]);
                    va = curve[g] + frac * (curve[g+1] - curve[g]);
                    break;
                }
            }
            double vb = curve[0];
            for (int g = 0; g < G; g++) {
                if (b_val <= grid_ptr[g+1]) {
                    double frac = (b_val - grid_ptr[g]) / (grid_ptr[g+1] - grid_ptr[g]);
                    vb = curve[g] + frac * (curve[g+1] - curve[g]);
                    break;
                }
            }

            dir_psi[d] = (va - vb) / (a_val - b_val);
        }

        psi_A_vec[r] = (dir_psi[0] + dir_psi[1]) / 2.0;
        psi_B_vec[r] = (dir_psi[2] + dir_psi[3]) / 2.0;
    }

    NumericVector psi_A(R), psi_B(R);
    for (int r = 0; r < R; r++) { psi_A[r] = psi_A_vec[r]; psi_B[r] = psi_B_vec[r]; }
    return List::create(Named("psi_A") = psi_A, Named("psi_B") = psi_B);
}


// ============================================================
// pasr_extract_all_marginal_cpp
//
// Batch marginalized prediction: given R replicates, n_queries
// counterfactual X matrices, and precomputed omega weights,
// returns R x n_queries matrix of marginalized predictions.
// Single C++ call eliminates 400+ R-to-C++ overhead.
// ============================================================

// [[Rcpp::export]]
NumericMatrix pasr_extract_all_marginal_cpp(
    List cache_list,           // R-length list of caches
    List Y_syn_list,           // R-length list of Y_syn vectors
    List X_cf_list,            // n_queries-length list of X_cf matrices (n x p, col-major)
    NumericMatrix omega_mat,   // n x n_queries matrix of omega weights
    int n_threads = 1
) {
    int R = cache_list.size();
    int n_queries = X_cf_list.size();

    // Pre-extract X_cf pointers
    std::vector<const double*> xcf_ptrs(n_queries);
    std::vector<int> xcf_nrow(n_queries);
    for (int q = 0; q < n_queries; q++) {
        NumericMatrix Xm = as<NumericMatrix>(X_cf_list[q]);
        xcf_ptrs[q] = REAL(Xm);
        xcf_nrow[q] = Xm.nrow();
    }
    const double* omega_ptr = REAL(omega_mat);
    int n_omega = omega_mat.nrow();

    // Pre-extract cache data
    struct CDat {
        int B, n, p, n_hon;
        const int* honest_idx;
        const int* tree_offsets;
        const int* flat_sv;
        const double* flat_sval;
        const int* flat_lc;
        const int* flat_rc;
        const double* leaf_mean_flat;
        const double* fhat_obs;
    };

    // 2 directions per replicate: A_AB, A_BA
    std::vector<std::vector<CDat>> all_cd(R, std::vector<CDat>(2));
    std::vector<std::vector<List>> holders(R, std::vector<List>(2));
    const char* dir_names[2] = {"A_AB", "A_BA"};

    // Pre-extract Y_syn pointers
    std::vector<const double*> ysyn_ptrs(R);
    std::vector<NumericVector> ysyn_holders(R);
    for (int r = 0; r < R; r++) {
        ysyn_holders[r] = as<NumericVector>(Y_syn_list[r]);
        ysyn_ptrs[r] = REAL(ysyn_holders[r]);
    }

    for (int r = 0; r < R; r++) {
        List rep = cache_list[r];
        for (int d = 0; d < 2; d++) {
            List c = rep[dir_names[d]];
            holders[r][d] = c;
            CDat& cd = all_cd[r][d];
            cd.B = as<int>(c["B"]);
            cd.n = as<int>(c["n"]);
            cd.p = as<int>(c["p"]);
            cd.n_hon = as<int>(c["n_hon"]);
            cd.honest_idx = INTEGER(as<IntegerVector>(c["honest_idx"]));
            cd.tree_offsets = INTEGER(as<IntegerVector>(c["tree_offsets"]));
            cd.flat_sv = INTEGER(as<IntegerVector>(c["flat_sv"]));
            cd.flat_sval = REAL(as<NumericVector>(c["flat_sval"]));
            cd.flat_lc = INTEGER(as<IntegerVector>(c["flat_lc"]));
            cd.flat_rc = INTEGER(as<IntegerVector>(c["flat_rc"]));
            cd.leaf_mean_flat = REAL(as<NumericVector>(c["leaf_mean_flat"]));
            cd.fhat_obs = REAL(as<NumericVector>(c["fhat_obs"]));
        }
    }

    // Output: R x n_queries
    std::vector<double> result_flat(R * n_queries, NA_REAL);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int r = 0; r < R; r++) {
        const double* Y_syn = ysyn_ptrs[r];

        for (int q = 0; q < n_queries; q++) {
            const double* Xcf = xcf_ptrs[q];
            int n_cf = xcf_nrow[q];
            double phi_sum = 0.0;
            int phi_cnt = 0;

            for (int d = 0; d < 2; d++) {
                const CDat& cd = all_cd[r][d];
                int B = cd.B, p = cd.p, n_hon = cd.n_hon; (void)p;
                int n_full = cd.n; (void)n_full;

                // Route all observations through trees with X_cf
                // Then compute AIPW for honest obs only
                for (int j = 0; j < n_hon; j++) {
                    int k = cd.honest_idx[j] - 1;  // 0-based obs index

                    // fhat_cf: route obs k through all trees with X_cf
                    double cf_sum = 0.0; int cf_cnt = 0;
                    for (int bb = 0; bb < B; bb++) {
                        int off = cd.tree_offsets[bb];
                        int node = 0;
                        while (cd.flat_lc[off + node] != 0 || cd.flat_rc[off + node] != 0) {
                            int var = cd.flat_sv[off + node];
                            double val = Xcf[k + n_cf * var];
                            node = (val <= cd.flat_sval[off + node]) ?
                                   cd.flat_lc[off + node] : cd.flat_rc[off + node];
                        }
                        double lm = cd.leaf_mean_flat[off + node];
                        if (lm != -1e308) { cf_sum += lm; cf_cnt++; }
                    }

                    double fhat_cf = (cf_cnt > 0) ? cf_sum / cf_cnt : NA_REAL;
                    double fhat_obs_j = cd.fhat_obs[j];
                    if (ISNA(fhat_cf) || ISNA(fhat_obs_j)) continue;

                    double R_k = Y_syn[k] - fhat_obs_j;
                    double omega_k = omega_ptr[k + n_omega * q];

                    phi_sum += fhat_cf + omega_k * R_k;
                    phi_cnt++;
                }
            }

            result_flat[r * n_queries + q] = (phi_cnt > 0) ? phi_sum / phi_cnt : NA_REAL;
        }
    }

    // Convert to R matrix
    NumericMatrix result(R, n_queries);
    for (int r = 0; r < R; r++)
        for (int q = 0; q < n_queries; q++)
            result(r, q) = result_flat[r * n_queries + q];
    return result;
}


// ============================================================
// pasr_extract_all_level_curve_cpp
//
// Batch level curve: given R replicates, returns R x (G+1) matrix.
// Each entry is the AIPW-adjusted level at that grid point.
// ============================================================

// [[Rcpp::export]]
NumericMatrix pasr_extract_all_level_curve_cpp(
    List cache_list, List Y_syn_list,
    NumericVector ghat, NumericVector x_var,
    int var_col, NumericVector grid_points,
    int n_threads = 1
) {
    int R = cache_list.size();
    int G_plus_1 = grid_points.size();
    const double* grid_ptr = REAL(grid_points);
    const double* g_ptr = REAL(ghat);
    const double* xv_ptr = REAL(x_var);

    struct CDatL {
        int B, n, p, n_hon;
        const int* honest_idx;
        const int* tree_offsets;
        const int* flat_sv;
        const double* flat_sval;
        const int* flat_lc;
        const int* flat_rc;
        const double* X_row_flat;
        const int* obs_leaf_flat;
        const double* leaf_mean_flat;
        const int* splits_on;
        const double* fhat_obs;
    };

    // Pre-extract: 2 dirs per replicate (A_AB, A_BA)
    std::vector<std::vector<CDatL>> all_cd(R, std::vector<CDatL>(2));
    std::vector<std::vector<List>> holders(R, std::vector<List>(2));
    const char* dir_names[2] = {"A_AB", "A_BA"};

    std::vector<const double*> ysyn_ptrs(R);
    std::vector<NumericVector> ysyn_holders(R);
    for (int r = 0; r < R; r++) {
        ysyn_holders[r] = as<NumericVector>(Y_syn_list[r]);
        ysyn_ptrs[r] = REAL(ysyn_holders[r]);
    }

    for (int r = 0; r < R; r++) {
        List rep = cache_list[r];
        for (int d = 0; d < 2; d++) {
            List c = rep[dir_names[d]];
            holders[r][d] = c;
            CDatL& cd = all_cd[r][d];
            cd.B = as<int>(c["B"]);
            cd.n = as<int>(c["n"]);
            cd.p = as<int>(c["p"]);
            cd.n_hon = as<int>(c["n_hon"]);
            cd.honest_idx = INTEGER(as<IntegerVector>(c["honest_idx"]));
            cd.tree_offsets = INTEGER(as<IntegerVector>(c["tree_offsets"]));
            cd.flat_sv = INTEGER(as<IntegerVector>(c["flat_sv"]));
            cd.flat_sval = REAL(as<NumericVector>(c["flat_sval"]));
            cd.flat_lc = INTEGER(as<IntegerVector>(c["flat_lc"]));
            cd.flat_rc = INTEGER(as<IntegerVector>(c["flat_rc"]));
            cd.X_row_flat = REAL(as<NumericVector>(c["X_row_flat"]));
            cd.obs_leaf_flat = INTEGER(as<IntegerVector>(c["obs_leaf_flat"]));
            cd.leaf_mean_flat = REAL(as<NumericVector>(c["leaf_mean_flat"]));
            cd.splits_on = INTEGER(as<IntegerVector>(c["splits_on"]));
            cd.fhat_obs = REAL(as<NumericVector>(c["fhat_obs"]));
        }
    }

    std::vector<double> result_flat(R * G_plus_1, NA_REAL);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int r = 0; r < R; r++) {
        const double* Y_syn = ysyn_ptrs[r];
        std::vector<double> mu_accum(G_plus_1, 0.0);

        for (int d = 0; d < 2; d++) {
            const CDatL& cd = all_cd[r][d];
            int B = cd.B, n = cd.n, p = cd.p, n_hon = cd.n_hon;

            // sigma2_ej
            double ss = 0.0;
            for (int j = 0; j < n_hon; j++) {
                int i = cd.honest_idx[j] - 1;
                double ej = xv_ptr[i] - g_ptr[i];
                ss += ej * ej;
            }
            double sigma2_ej = ss / n_hon;

            // Grid accumulation: flat, tree-outer
            std::vector<double> gsum(G_plus_1 * n, 0.0);
            std::vector<int> gcnt(G_plus_1 * n, 0);

            for (int bb = 0; bb < B; bb++) {
                int off = cd.tree_offsets[bb];
                const int* sv = cd.flat_sv + off;
                const double* sval = cd.flat_sval + off;
                const int* lc = cd.flat_lc + off;
                const int* rc = cd.flat_rc + off;
                const double* lm = cd.leaf_mean_flat + off;
                int leaf_off = bb * n_hon;
                bool tree_has_split = (cd.splits_on[var_col * B + bb] != 0);

                if (tree_has_split) {
                    for (int j = 0; j < n_hon; j++) {
                        int i = cd.honest_idx[j] - 1;
                        int xbase = j * p;

                        // Shared-prefix
                        int prefix_node = 0;
                        while (lc[prefix_node] != 0 || rc[prefix_node] != 0) {
                            if (sv[prefix_node] == var_col) break;
                            prefix_node = (cd.X_row_flat[xbase + sv[prefix_node]] <= sval[prefix_node]) ?
                                          lc[prefix_node] : rc[prefix_node];
                        }
                        if (lc[prefix_node] == 0 && rc[prefix_node] == 0) {
                            if (lm[prefix_node] != -1e308) {
                                double v = lm[prefix_node];
                                for (int g = 0; g < G_plus_1; g++) {
                                    gsum[g * n + i] += v; gcnt[g * n + i]++;
                                }
                            }
                        } else {
                            for (int g = 0; g < G_plus_1; g++) {
                                double gval = grid_ptr[g];
                                int node = (gval <= sval[prefix_node]) ?
                                           lc[prefix_node] : rc[prefix_node];
                                while (lc[node] != 0 || rc[node] != 0) {
                                    double xv = (sv[node] == var_col) ? gval :
                                                cd.X_row_flat[xbase + sv[node]];
                                    node = (xv <= sval[node]) ? lc[node] : rc[node];
                                }
                                if (lm[node] != -1e308) {
                                    gsum[g * n + i] += lm[node]; gcnt[g * n + i]++;
                                }
                            }
                        }
                    }
                } else {
                    for (int j = 0; j < n_hon; j++) {
                        int i = cd.honest_idx[j] - 1;
                        int leaf = cd.obs_leaf_flat[leaf_off + j];
                        if (lm[leaf] != -1e308) {
                            double v = lm[leaf];
                            for (int g = 0; g < G_plus_1; g++) {
                                gsum[g * n + i] += v; gcnt[g * n + i]++;
                            }
                        }
                    }
                }
            }

            // AIPW at each grid point
            for (int g = 0; g < G_plus_1; g++) {
                double phi_sum = 0.0; int cnt = 0;
                for (int j = 0; j < n_hon; j++) {
                    int i = cd.honest_idx[j] - 1;
                    double fg = (gcnt[g * n + i] > 0) ? gsum[g * n + i] / gcnt[g * n + i] : NA_REAL;
                    double fo = cd.fhat_obs[j];
                    if (ISNA(fg) || ISNA(fo)) continue;
                    double omega = (xv_ptr[i] - g_ptr[i]) / sigma2_ej;
                    double R_k = Y_syn[i] - fo;
                    phi_sum += fg + omega * R_k;
                    cnt++;
                }
                mu_accum[g] += (cnt > 0) ? phi_sum / cnt : 0.0;
            }
        }

        // Average over 2 directions
        for (int g = 0; g < G_plus_1; g++)
            result_flat[r * G_plus_1 + g] = mu_accum[g] / 2.0;
    }

    NumericMatrix result(R, G_plus_1);
    for (int r = 0; r < R; r++)
        for (int g = 0; g < G_plus_1; g++)
            result(r, g) = result_flat[r * G_plus_1 + g];
    return result;
}


// ============================================================
// ranger_batch_predict_cpp
//
// Pre-extracts N ranger forests into flat arrays, then predicts
// all of them at query points in one call. Returns either:
// - For conditional: nk x N matrix of predictions (one per forest)
// - For unconditional: nk x M matrix of f_bar (averaged over K per bootstrap)
//
// forests_list: list of stripped ranger forest objects
// X_query: nk x p matrix of query points (column-major)
// group_sizes: integer vector of length M, each entry = K (forests per group)
//              If NULL, no grouping — return raw predictions
// outcome_binary: if TRUE, extract column 2 of probability predictions
// n_threads: OpenMP threads
// ============================================================

// [[Rcpp::export]]
NumericMatrix ranger_batch_predict_cpp(
    List forests_list, NumericMatrix X_query,
    Nullable<IntegerVector> group_sizes_ = R_NilValue,
    bool outcome_binary = false,
    int n_threads = 1
) {
    int N = forests_list.size();
    int nk = X_query.nrow();
    int p = X_query.ncol(); (void)p;
    const double* Xq_ptr = REAL(X_query);

    // Pre-extract all forests into flat arrays
    struct FlatForest {
        int B;
        std::vector<int> tree_offsets;
        std::vector<int> flat_sv;
        std::vector<double> flat_sval;
        std::vector<int> flat_lc;
        std::vector<int> flat_rc;
        // For regression: leaf values indexed by node
        // For probability: leaf prob (class 2) indexed by node
        std::vector<double> leaf_val;
    };

    std::vector<FlatForest> all_forests(N);

    for (int f = 0; f < N; f++) {
        List rf = forests_list[f];
        List forest = rf["forest"];
        List svl = forest["split.varIDs"];
        List svall = forest["split.values"];
        List chl = forest["child.nodeIDs"];
        int B = svl.size();
        all_forests[f].B = B;
        all_forests[f].tree_offsets.resize(B);

        int total_nodes = 0;
        for (int b = 0; b < B; b++) {
            all_forests[f].tree_offsets[b] = total_nodes;
            IntegerVector sv_r = svl[b];
            total_nodes += sv_r.size();
        }

        all_forests[f].flat_sv.resize(total_nodes);
        all_forests[f].flat_sval.resize(total_nodes);
        all_forests[f].flat_lc.resize(total_nodes);
        all_forests[f].flat_rc.resize(total_nodes);
        all_forests[f].leaf_val.resize(total_nodes, 0.0);

        for (int b = 0; b < B; b++) {
            int off = all_forests[f].tree_offsets[b];
            IntegerVector sv_r = svl[b];
            NumericVector sval_r = svall[b];
            List ch = chl[b];
            IntegerVector lc_r = ch[0], rc_r = ch[1];
            int nn = sv_r.size();
            for (int nd = 0; nd < nn; nd++) {
                all_forests[f].flat_sv[off + nd] = sv_r[nd];
                all_forests[f].flat_sval[off + nd] = sval_r[nd];
                all_forests[f].flat_lc[off + nd] = lc_r[nd];
                all_forests[f].flat_rc[off + nd] = rc_r[nd];
            }
        }

        // Extract leaf predictions
        if (outcome_binary) {
            // Probability forest: terminal.class.counts stores per-node class proportions
            if (forest.containsElementNamed("terminal.class.counts")) {
                List tcc = forest["terminal.class.counts"];
                for (int b = 0; b < B; b++) {
                    int off = all_forests[f].tree_offsets[b];
                    NumericMatrix counts_b = as<NumericMatrix>(tcc[b]);
                    int nn = counts_b.nrow();
                    for (int nd = 0; nd < nn; nd++) {
                        double total = 0;
                        for (int c = 0; c < counts_b.ncol(); c++) total += counts_b(nd, c);
                        all_forests[f].leaf_val[off + nd] =
                            (total > 0) ? counts_b(nd, 1) / total : 0.0;
                    }
                }
            }
        } else {
            // Regression forest: leaf nodes store prediction in split.values
            // Already copied to flat_sval — use directly
            for (int nd = 0; nd < total_nodes; nd++) {
                all_forests[f].leaf_val[nd] = all_forests[f].flat_sval[nd];
            }
        }
    }

    // Determine output shape
    bool has_groups = group_sizes_.isNotNull();
    int n_groups = 0;
    std::vector<int> grp_start, grp_size;
    if (has_groups) {
        IntegerVector gs = as<IntegerVector>(group_sizes_);
        n_groups = gs.size();
        grp_start.resize(n_groups);
        grp_size.resize(n_groups);
        int pos = 0;
        for (int g = 0; g < n_groups; g++) {
            grp_start[g] = pos;
            grp_size[g] = gs[g];
            pos += gs[g];
        }
    }

    int n_cols = has_groups ? n_groups : N;
    std::vector<double> result_flat(nk * n_cols, 0.0);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    if (has_groups) {
        // Grouped: compute f_bar_m = (1/K) sum_k f_hat_{m,k}(x)
        #pragma omp parallel for schedule(static) if(n_threads > 1)
        for (int g = 0; g < n_groups; g++) {
            int start = grp_start[g];
            int K = grp_size[g];
            for (int q = 0; q < nk; q++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    int fi = start + k;
                    const FlatForest& ff = all_forests[fi];
                    double tree_sum = 0.0;
                    int tree_cnt = 0;
                    for (int b = 0; b < ff.B; b++) {
                        int off = ff.tree_offsets[b];
                        int node = 0;
                        while (ff.flat_lc[off + node] != 0 || ff.flat_rc[off + node] != 0) {
                            int var = ff.flat_sv[off + node];
                            double val = Xq_ptr[q + nk * var];
                            node = (val <= ff.flat_sval[off + node]) ?
                                   ff.flat_lc[off + node] : ff.flat_rc[off + node];
                        }
                        tree_sum += ff.leaf_val[off + node];
                        tree_cnt++;
                    }
                    sum += (tree_cnt > 0) ? tree_sum / tree_cnt : 0.0;
                }
                result_flat[q + nk * g] = sum / K;
            }
        }
    } else {
        // Ungrouped: raw predictions per forest
        #pragma omp parallel for schedule(static) if(n_threads > 1)
        for (int fi = 0; fi < N; fi++) {
            const FlatForest& ff = all_forests[fi];
            for (int q = 0; q < nk; q++) {
                double tree_sum = 0.0;
                int tree_cnt = 0;
                for (int b = 0; b < ff.B; b++) {
                    int off = ff.tree_offsets[b];
                    int node = 0;
                    while (ff.flat_lc[off + node] != 0 || ff.flat_rc[off + node] != 0) {
                        int var = ff.flat_sv[off + node];
                        double val = Xq_ptr[q + nk * var];
                        node = (val <= ff.flat_sval[off + node]) ?
                               ff.flat_lc[off + node] : ff.flat_rc[off + node];
                    }
                    tree_sum += ff.leaf_val[off + node];
                    tree_cnt++;
                }
                result_flat[q + nk * fi] = (tree_cnt > 0) ? tree_sum / tree_cnt : NA_REAL;
            }
        }
    }

    NumericMatrix result(nk, n_cols);
    for (int j = 0; j < n_cols; j++)
        for (int q = 0; q < nk; q++)
            result(q, j) = result_flat[q + nk * j];
    return result;
}


// ============================================================
// preextract_ranger_forests_cpp
//
// Pre-extracts N ranger forests into flat arrays at fit time.
// Returns a cache list that ranger_batch_predict_cached_cpp uses
// at predict time with zero R-level extraction overhead.
// ============================================================

// [[Rcpp::export]]
List preextract_ranger_forests_cpp(List forests_list, bool outcome_binary = false) {
    int N = forests_list.size();

    // Collect all flat arrays
    std::vector<int> all_B(N);
    std::vector<int> forest_node_offsets(N);  // offset into the global flat arrays
    int total_nodes_all = 0;

    // First pass: count total nodes
    for (int f = 0; f < N; f++) {
        List rf = forests_list[f];
        List forest = rf["forest"];
        List svl = forest["split.varIDs"];
        int B = svl.size();
        all_B[f] = B;
        forest_node_offsets[f] = total_nodes_all;
        for (int b = 0; b < B; b++) {
            IntegerVector sv_r = svl[b];
            total_nodes_all += sv_r.size();
        }
    }

    // Global flat arrays
    IntegerVector g_sv(total_nodes_all);
    NumericVector g_sval(total_nodes_all);
    IntegerVector g_lc(total_nodes_all);
    IntegerVector g_rc(total_nodes_all);
    NumericVector g_leafval(total_nodes_all);

    // Per-forest: tree offsets within that forest's node block
    // We need: for forest f, tree b, the offset is forest_node_offsets[f] + local_tree_offset[f][b]
    // Store as flat: tree_offsets_flat[sum(B[0..f-1]) + b] = global node offset for forest f, tree b
    int total_trees = 0;
    for (int f = 0; f < N; f++) total_trees += all_B[f];
    IntegerVector tree_offsets_flat(total_trees);
    IntegerVector forest_tree_start(N);  // where each forest's tree offsets begin

    int tree_idx = 0;
    for (int f = 0; f < N; f++) {
        forest_tree_start[f] = tree_idx;
        List rf = forests_list[f];
        List forest = rf["forest"];
        List svl = forest["split.varIDs"];
        List svall = forest["split.values"];
        List chl = forest["child.nodeIDs"];
        int B = all_B[f];
        int goff = forest_node_offsets[f];
        int local_off = 0;

        for (int b = 0; b < B; b++) {
            tree_offsets_flat[tree_idx] = goff + local_off;
            tree_idx++;

            IntegerVector sv_r = svl[b];
            NumericVector sval_r = svall[b];
            List ch = chl[b];
            IntegerVector lc_r = ch[0], rc_r = ch[1];
            int nn = sv_r.size();

            for (int nd = 0; nd < nn; nd++) {
                int gi = goff + local_off + nd;
                g_sv[gi] = sv_r[nd];
                g_sval[gi] = sval_r[nd];
                g_lc[gi] = lc_r[nd];
                g_rc[gi] = rc_r[nd];
            }
            local_off += nn;
        }

        // Leaf values
        if (outcome_binary) {
            if (forest.containsElementNamed("terminal.class.counts")) {
                List tcc = forest["terminal.class.counts"];
                local_off = 0;
                for (int b = 0; b < B; b++) {
                    NumericMatrix counts_b = as<NumericMatrix>(tcc[b]);
                    int nn = counts_b.nrow();
                    for (int nd = 0; nd < nn; nd++) {
                        double total = 0;
                        for (int c = 0; c < counts_b.ncol(); c++) total += counts_b(nd, c);
                        g_leafval[goff + local_off + nd] =
                            (total > 0) ? counts_b(nd, 1) / total : 0.0;
                    }
                    local_off += nn;
                }
            }
        } else {
            for (int nd = 0; nd < (int)(local_off); nd++) {
                g_leafval[goff + nd] = g_sval[goff + nd];
            }
        }
    }

    return List::create(
        Named("N") = N,
        Named("B") = IntegerVector(all_B.begin(), all_B.end()),
        Named("forest_tree_start") = forest_tree_start,
        Named("tree_offsets") = tree_offsets_flat,
        Named("flat_sv") = g_sv,
        Named("flat_sval") = g_sval,
        Named("flat_lc") = g_lc,
        Named("flat_rc") = g_rc,
        Named("leaf_val") = g_leafval,
        Named("total_nodes") = total_nodes_all,
        Named("total_trees") = total_trees
    );
}


// ============================================================
// ranger_predict_from_cache_cpp
//
// Predicts from pre-extracted cache. No R list access at predict time.
// group_sizes: if provided, returns nk x M matrix of f_bar (averaged over K)
//              if NULL, returns nk x N matrix of raw predictions
// ============================================================

// [[Rcpp::export]]
NumericMatrix ranger_predict_from_cache_cpp(
    List cache, NumericMatrix X_query,
    Nullable<IntegerVector> group_sizes_ = R_NilValue,
    int n_threads = 1
) {
    int N = as<int>(cache["N"]);
    IntegerVector B_vec = as<IntegerVector>(cache["B"]);
    IntegerVector fts = as<IntegerVector>(cache["forest_tree_start"]);
    IntegerVector toff = as<IntegerVector>(cache["tree_offsets"]);
    IntegerVector fsv = as<IntegerVector>(cache["flat_sv"]);
    NumericVector fsval = as<NumericVector>(cache["flat_sval"]);
    IntegerVector flc = as<IntegerVector>(cache["flat_lc"]);
    IntegerVector frc = as<IntegerVector>(cache["flat_rc"]);
    NumericVector lv = as<NumericVector>(cache["leaf_val"]);

    const int* sv_ptr = INTEGER(fsv);
    const double* sval_ptr = REAL(fsval);
    const int* lc_ptr = INTEGER(flc);
    const int* rc_ptr = INTEGER(frc);
    const double* lv_ptr = REAL(lv);
    const int* toff_ptr = INTEGER(toff);
    const int* fts_ptr = INTEGER(fts);
    const int* B_ptr = INTEGER(B_vec);

    int nk = X_query.nrow();
    const double* Xq_ptr = REAL(X_query);

    bool has_groups = group_sizes_.isNotNull();
    int n_groups = 0;
    std::vector<int> grp_start, grp_size;
    if (has_groups) {
        IntegerVector gs = as<IntegerVector>(group_sizes_);
        n_groups = gs.size();
        grp_start.resize(n_groups);
        grp_size.resize(n_groups);
        int pos = 0;
        for (int g = 0; g < n_groups; g++) {
            grp_start[g] = pos;
            grp_size[g] = gs[g];
            pos += gs[g];
        }
    }

    int n_cols = has_groups ? n_groups : N;
    std::vector<double> result_flat(nk * n_cols, 0.0);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    if (has_groups) {
        #pragma omp parallel for schedule(static) if(n_threads > 1)
        for (int g = 0; g < n_groups; g++) {
            int start = grp_start[g];
            int K = grp_size[g];
            for (int q = 0; q < nk; q++) {
                double sum = 0.0;
                for (int ki = 0; ki < K; ki++) {
                    int fi = start + ki;
                    int B = B_ptr[fi];
                    int ts = fts_ptr[fi];
                    double tree_sum = 0.0;
                    for (int b = 0; b < B; b++) {
                        int off = toff_ptr[ts + b];
                        int node = 0;
                        while (lc_ptr[off + node] != 0 || rc_ptr[off + node] != 0) {
                            int var = sv_ptr[off + node];
                            double val = Xq_ptr[q + nk * var];
                            node = (val <= sval_ptr[off + node]) ?
                                   lc_ptr[off + node] : rc_ptr[off + node];
                        }
                        tree_sum += lv_ptr[off + node];
                    }
                    sum += tree_sum / B;
                }
                result_flat[q + nk * g] = sum / K;
            }
        }
    } else {
        #pragma omp parallel for schedule(static) if(n_threads > 1)
        for (int fi = 0; fi < N; fi++) {
            int B = B_ptr[fi];
            int ts = fts_ptr[fi];
            for (int q = 0; q < nk; q++) {
                double tree_sum = 0.0;
                for (int b = 0; b < B; b++) {
                    int off = toff_ptr[ts + b];
                    int node = 0;
                    while (lc_ptr[off + node] != 0 || rc_ptr[off + node] != 0) {
                        int var = sv_ptr[off + node];
                        double val = Xq_ptr[q + nk * var];
                        node = (val <= sval_ptr[off + node]) ?
                               lc_ptr[off + node] : rc_ptr[off + node];
                    }
                    tree_sum += lv_ptr[off + node];
                }
                result_flat[q + nk * fi] = tree_sum / B;
            }
        }
    }

    NumericMatrix result(nk, n_cols);
    for (int j = 0; j < n_cols; j++)
        for (int q = 0; q < nk; q++)
            result(q, j) = result_flat[q + nk * j];
    return result;
}


// ============================================================
// flatten_ranger_forest_cpp
//
// Extracts a ranger forest into minimal flat arrays for
// fast batch prediction. No honest splitting, no obs routing.
// Just tree structure + leaf predictions.
// ============================================================

// [[Rcpp::export]]
List flatten_ranger_forest_cpp(List rf, bool outcome_binary = false) {
    List forest = rf["forest"];
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();

    // Count total nodes
    int total_nodes = 0;
    std::vector<int> tree_offsets(B);
    for (int b = 0; b < B; b++) {
        tree_offsets[b] = total_nodes;
        IntegerVector sv_r = svl[b];
        total_nodes += sv_r.size();
    }

    IntegerVector flat_sv(total_nodes);
    NumericVector flat_sval(total_nodes);
    IntegerVector flat_lc(total_nodes);
    IntegerVector flat_rc(total_nodes);
    NumericVector leaf_val(total_nodes, 0.0);

    for (int b = 0; b < B; b++) {
        int off = tree_offsets[b];
        IntegerVector sv_r = svl[b];
        NumericVector sval_r = svall[b];
        List ch = chl[b];
        IntegerVector lc_r = ch[0], rc_r = ch[1];
        int nn = sv_r.size();
        for (int nd = 0; nd < nn; nd++) {
            flat_sv[off + nd] = sv_r[nd];
            flat_sval[off + nd] = sval_r[nd];
            flat_lc[off + nd] = lc_r[nd];
            flat_rc[off + nd] = rc_r[nd];
        }
    }

    // Leaf values
    if (outcome_binary && forest.containsElementNamed("terminal.class.counts")) {
        List tcc = forest["terminal.class.counts"];
        for (int b = 0; b < B; b++) {
            int off = tree_offsets[b];
            NumericMatrix counts_b = as<NumericMatrix>(tcc[b]);
            int nn = counts_b.nrow();
            for (int nd = 0; nd < nn; nd++) {
                double total = 0;
                for (int c = 0; c < counts_b.ncol(); c++) total += counts_b(nd, c);
                leaf_val[off + nd] = (total > 0) ? counts_b(nd, 1) / total : 0.0;
            }
        }
    } else {
        // Regression: leaf predictions stored in split.values at leaf nodes
        for (int nd = 0; nd < total_nodes; nd++) {
            leaf_val[nd] = flat_sval[nd];
        }
    }

    IntegerVector tree_off_r(B);
    for (int b = 0; b < B; b++) tree_off_r[b] = tree_offsets[b];

    return List::create(
        Named("B") = B,
        Named("tree_offsets") = tree_off_r,
        Named("flat_sv") = flat_sv,
        Named("flat_sval") = flat_sval,
        Named("flat_lc") = flat_lc,
        Named("flat_rc") = flat_rc,
        Named("leaf_val") = leaf_val
    );
}


// ============================================================
// predict_flat_forest_cpp
//
// Predicts from a single flattened forest at query points.
// ============================================================

// [[Rcpp::export]]
NumericVector predict_flat_forest_cpp(
    List flat_cache, NumericMatrix X_query, int n_threads = 1
) {
    int B = as<int>(flat_cache["B"]);
    IntegerVector tree_offsets = as<IntegerVector>(flat_cache["tree_offsets"]);
    IntegerVector flat_sv = as<IntegerVector>(flat_cache["flat_sv"]);
    NumericVector flat_sval = as<NumericVector>(flat_cache["flat_sval"]);
    IntegerVector flat_lc = as<IntegerVector>(flat_cache["flat_lc"]);
    IntegerVector flat_rc = as<IntegerVector>(flat_cache["flat_rc"]);
    NumericVector leaf_val = as<NumericVector>(flat_cache["leaf_val"]);

    int nk = X_query.nrow();
    const double* Xq = REAL(X_query);
    const int* off_arr = INTEGER(tree_offsets);
    const int* sv = INTEGER(flat_sv);
    const double* sval = REAL(flat_sval);
    const int* lc = INTEGER(flat_lc);
    const int* rc = INTEGER(flat_rc);
    const double* lv = REAL(leaf_val);

    NumericVector result(nk);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int q = 0; q < nk; q++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            int off = off_arr[b];
            int node = 0;
            while (lc[off + node] != 0 || rc[off + node] != 0) {
                int var = sv[off + node];
                double val = Xq[q + nk * var];
                node = (val <= sval[off + node]) ? lc[off + node] : rc[off + node];
            }
            sum += lv[off + node];
        }
        result[q] = sum / B;
    }
    return result;
}


// ============================================================
// batch_predict_flat_forests_cpp
//
// Batch predict from N pre-flattened forests.
// Optional grouping: returns group means (f_bar per bootstrap).
// ============================================================

// [[Rcpp::export]]
NumericMatrix batch_predict_flat_forests_cpp(
    List flat_caches,
    NumericMatrix X_query,
    Nullable<IntegerVector> group_sizes_ = R_NilValue,
    int n_threads = 1
) {
    int N = flat_caches.size();
    int nk = X_query.nrow();
    const double* Xq = REAL(X_query);

    // Pre-extract all flat caches
    struct FF {
        int B;
        const int* off;
        const int* sv;
        const double* sval;
        const int* lc;
        const int* rc;
        const double* lv;
    };

    std::vector<FF> all_ff(N);
    // Keep Rcpp objects alive
    std::vector<List> holders(N);
    for (int f = 0; f < N; f++) {
        List fc = flat_caches[f];
        holders[f] = fc;
        FF& ff = all_ff[f];
        ff.B = as<int>(fc["B"]);
        ff.off = INTEGER(as<IntegerVector>(fc["tree_offsets"]));
        ff.sv = INTEGER(as<IntegerVector>(fc["flat_sv"]));
        ff.sval = REAL(as<NumericVector>(fc["flat_sval"]));
        ff.lc = INTEGER(as<IntegerVector>(fc["flat_lc"]));
        ff.rc = INTEGER(as<IntegerVector>(fc["flat_rc"]));
        ff.lv = REAL(as<NumericVector>(fc["leaf_val"]));
    }

    bool has_groups = group_sizes_.isNotNull();
    int n_groups = 0;
    std::vector<int> grp_start, grp_size;
    if (has_groups) {
        IntegerVector gs = as<IntegerVector>(group_sizes_);
        n_groups = gs.size();
        grp_start.resize(n_groups);
        grp_size.resize(n_groups);
        int pos = 0;
        for (int g = 0; g < n_groups; g++) {
            grp_start[g] = pos;
            grp_size[g] = gs[g];
            pos += gs[g];
        }
    }

    int n_cols = has_groups ? n_groups : N;
    std::vector<double> result_flat(nk * n_cols, 0.0);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    if (has_groups) {
        #pragma omp parallel for schedule(static) if(n_threads > 1)
        for (int g = 0; g < n_groups; g++) {
            int start = grp_start[g];
            int K = grp_size[g];
            for (int q = 0; q < nk; q++) {
                double group_sum = 0.0;
                for (int k = 0; k < K; k++) {
                    const FF& ff = all_ff[start + k];
                    double tree_sum = 0.0;
                    for (int b = 0; b < ff.B; b++) {
                        int off = ff.off[b];
                        int node = 0;
                        while (ff.lc[off + node] != 0 || ff.rc[off + node] != 0) {
                            int var = ff.sv[off + node];
                            double val = Xq[q + nk * var];
                            node = (val <= ff.sval[off + node]) ?
                                   ff.lc[off + node] : ff.rc[off + node];
                        }
                        tree_sum += ff.lv[off + node];
                    }
                    group_sum += tree_sum / ff.B;
                }
                result_flat[q + nk * g] = group_sum / K;
            }
        }
    } else {
        #pragma omp parallel for schedule(static) if(n_threads > 1)
        for (int fi = 0; fi < N; fi++) {
            const FF& ff = all_ff[fi];
            for (int q = 0; q < nk; q++) {
                double tree_sum = 0.0;
                for (int b = 0; b < ff.B; b++) {
                    int off = ff.off[b];
                    int node = 0;
                    while (ff.lc[off + node] != 0 || ff.rc[off + node] != 0) {
                        int var = ff.sv[off + node];
                        double val = Xq[q + nk * var];
                        node = (val <= ff.sval[off + node]) ?
                               ff.lc[off + node] : ff.rc[off + node];
                    }
                    tree_sum += ff.lv[off + node];
                }
                result_flat[q + nk * fi] = tree_sum / ff.B;
            }
        }
    }

    NumericMatrix result(nk, n_cols);
    for (int j = 0; j < n_cols; j++)
        for (int q = 0; q < nk; q++)
            result(q, j) = result_flat[q + nk * j];
    return result;
}


// ============================================================
// predict_flat_forest_all_cpp
//
// Returns per-tree predictions from a flattened forest.
// Output: nk x B matrix (each column = one tree's prediction)
// Used for computing mc_var = Var(trees) / B
// ============================================================

// [[Rcpp::export]]
NumericMatrix predict_flat_forest_all_cpp(
    List flat_cache, NumericMatrix X_query, int n_threads = 1
) {
    int B = as<int>(flat_cache["B"]);
    IntegerVector tree_offsets = as<IntegerVector>(flat_cache["tree_offsets"]);
    IntegerVector flat_sv = as<IntegerVector>(flat_cache["flat_sv"]);
    NumericVector flat_sval = as<NumericVector>(flat_cache["flat_sval"]);
    IntegerVector flat_lc = as<IntegerVector>(flat_cache["flat_lc"]);
    IntegerVector flat_rc = as<IntegerVector>(flat_cache["flat_rc"]);
    NumericVector leaf_val = as<NumericVector>(flat_cache["leaf_val"]);

    int nk = X_query.nrow();
    const double* Xq = REAL(X_query);
    const int* off_arr = INTEGER(tree_offsets);
    const int* sv = INTEGER(flat_sv);
    const double* sval = REAL(flat_sval);
    const int* lc = INTEGER(flat_lc);
    const int* rc = INTEGER(flat_rc);
    const double* lv = REAL(leaf_val);

    // nk x B output
    std::vector<double> result_flat(nk * B);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int q = 0; q < nk; q++) {
        for (int b = 0; b < B; b++) {
            int off = off_arr[b];
            int node = 0;
            while (lc[off + node] != 0 || rc[off + node] != 0) {
                int var = sv[off + node];
                double val = Xq[q + nk * var];
                node = (val <= sval[off + node]) ? lc[off + node] : rc[off + node];
            }
            result_flat[q + nk * b] = lv[off + node];
        }
    }

    NumericMatrix result(nk, B);
    for (int b = 0; b < B; b++)
        for (int q = 0; q < nk; q++)
            result(q, b) = result_flat[q + nk * b];
    return result;
}
#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// ============================================================
// compute_design_point_variance_cpp
//
// Computes V_X(x) = Var_w(f_hat(X_k)) / n_eff(x) for each query point.
// Direct weighted sample variance of forest predictions at training
// neighbors — captures ALL orders of variation, no Taylor expansion.
//
// Inputs:
//   forest: ranger forest list
//   X_train: n x p (unused but kept for signature compatibility)
//   X_query: nk x p query matrix
//   train_leaf_ids: n x B integer matrix (terminal node IDs)
//   inbag: n x B integer matrix (>0 if in-bag)
//   f_hat_train: n-vector — deployed forest predictions at training points
//   f_hat_query: nk-vector — deployed forest predictions at query points (unused)
//   outcome_binary: bool
//   n_threads: OpenMP threads
//
// Algorithm per query point:
//   1. Route query through B trees → leaf IDs
//   2. Accumulate weights w_k from in-bag obs sharing leaves
//   3. n_eff = 1 / sum(w_k^2)
//   4. f_w = sum_k w_k * f_hat_train[k]  (weighted mean prediction)
//   5. V_X = sum_k w_k * (f_hat_train[k] - f_w)^2 / n_eff
// ============================================================

// [[Rcpp::export]]
List compute_design_point_variance_cpp(
    List forest,
    NumericMatrix X_query,
    IntegerMatrix train_leaf_ids,
    IntegerMatrix inbag,
    NumericVector f_hat_train,
    bool outcome_binary,
    int n_threads
) {
    // Extract forest structure
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = train_leaf_ids.nrow();
    int nk = X_query.nrow();

    // Pre-extract forest into flat arrays (sequential)
    int total_nodes = 0;
    std::vector<int> tree_off(B);
    for (int b = 0; b < B; b++) {
        tree_off[b] = total_nodes;
        IntegerVector sv = svl[b];
        total_nodes += sv.size();
    }
    std::vector<int> flat_sv(total_nodes);
    std::vector<double> flat_sval(total_nodes);
    std::vector<int> flat_lc(total_nodes);
    std::vector<int> flat_rc(total_nodes);

    for (int b = 0; b < B; b++) {
        int off = tree_off[b];
        IntegerVector sv = svl[b];
        NumericVector sval = svall[b];
        List ch = chl[b];
        IntegerVector lc_r = ch[0], rc_r = ch[1];
        int nn = sv.size();
        for (int nd = 0; nd < nn; nd++) {
            flat_sv[off + nd] = sv[nd];
            flat_sval[off + nd] = sval[nd];
            flat_lc[off + nd] = lc_r[nd];
            flat_rc[off + nd] = rc_r[nd];
        }
    }

    // Pre-extract raw pointers
    const double* Xq = REAL(X_query);
    const int* tli = INTEGER(train_leaf_ids);
    const int* inb = INTEGER(inbag);
    const double* fht = REAL(f_hat_train);

    // Output
    NumericVector V_X(nk);
    NumericVector n_eff_out(nk);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int q = 0; q < nk; q++) {
        // --- Step 1: Route query through all trees ---
        std::vector<int> q_leaf(B);
        for (int b = 0; b < B; b++) {
            int off = tree_off[b];
            int node = 0;
            while (flat_lc[off + node] != 0 || flat_rc[off + node] != 0) {
                int var = flat_sv[off + node];
                double val = Xq[q + nk * var];
                node = (val <= flat_sval[off + node]) ?
                       flat_lc[off + node] : flat_rc[off + node];
            }
            q_leaf[b] = node;
        }

        // --- Step 2: Accumulate weights (in-bag only) ---
        std::vector<double> w(n, 0.0);
        for (int b = 0; b < B; b++) {
            int ql = q_leaf[b];
            int leaf_count = 0;
            for (int i = 0; i < n; i++) {
                if (inb[i + n * b] > 0 && tli[i + n * b] == ql) leaf_count++;
            }
            if (leaf_count == 0) continue;
            double wt = 1.0 / ((double)B * (double)leaf_count);
            for (int i = 0; i < n; i++) {
                if (inb[i + n * b] > 0 && tli[i + n * b] == ql) w[i] += wt;
            }
        }

        // --- Step 3: n_eff ---
        double sum_w2 = 0.0;
        for (int i = 0; i < n; i++) sum_w2 += w[i] * w[i];
        double neff = (sum_w2 > 0) ? 1.0 / sum_w2 : 1.0;
        n_eff_out[q] = neff;

        // --- Step 4: Weighted mean of f_hat at neighbors ---
        double f_w = 0.0;
        for (int i = 0; i < n; i++) f_w += w[i] * fht[i];

        // --- Step 5: Weighted variance of f_hat at neighbors / n_eff ---
        double wvar = 0.0;
        for (int i = 0; i < n; i++) {
            if (w[i] <= 0) continue;
            double diff = fht[i] - f_w;
            wvar += w[i] * diff * diff;
        }

        V_X[q] = std::max(wvar / neff, 0.0);
    }

    return List::create(
        Named("V_X") = V_X,
        Named("n_eff") = n_eff_out
    );
}
