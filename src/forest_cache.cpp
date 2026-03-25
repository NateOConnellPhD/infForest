// forest_cache.cpp — Forest cache precomputation and cached AIPW scoring
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
