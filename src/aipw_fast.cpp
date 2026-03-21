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
    Nullable<NumericVector> indicator_ = R_NilValue
) {
    int B = as<int>(cache["B"]);
    int n = as<int>(cache["n"]);
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

    bool has_indicator = indicator_.isNotNull();
    NumericVector indicator_vec;
    const double* ind_ptr = nullptr;
    if (has_indicator) {
        indicator_vec = as<NumericVector>(indicator_);
        ind_ptr = REAL(indicator_vec);
    }

    // Per-observation accumulators (only a and b — obs is precomputed)
    std::vector<double> fhat_a_sum(n, 0.0), fhat_b_sum(n, 0.0);
    std::vector<int> fhat_a_cnt(n, 0), fhat_b_cnt(n, 0);

    int n_split_trees = 0;
    int n_diff_leaves = 0;  // debug: count obs-tree combos where a != b

    for (int bb = 0; bb < B; bb++) {
        int off = tree_offsets[bb];
        int nn = tree_nnodes[bb];
        const int* sv = INTEGER(flat_sv) + off;
        const double* sval = REAL(flat_sval) + off;
        const int* lc = INTEGER(flat_lc) + off;
        const int* rc = INTEGER(flat_rc) + off;
        const double* lm = REAL(leaf_mean_flat) + off;
        int leaf_off = bb * n_hon;

        bool tree_has_split = (splits_on[var_col * B + bb] != 0);
        if (tree_has_split) n_split_trees++;

        if (tree_has_split) {
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1;
                int lo = obs_leaf_flat[leaf_off + j];

                // Shared-prefix routing: walk once until var_col split, then branch
                int xbase = j * p;
                int node = 0;
                while (lc[node] != 0 || rc[node] != 0) {
                    if (sv[node] == var_col) break;
                    node = (X_ptr[xbase + sv[node]] <= sval[node]) ? lc[node] : rc[node];
                }

                int leaf_a_node, leaf_b_node;
                if (lc[node] == 0 && rc[node] == 0) {
                    // No var_col split on path — same leaf for both
                    leaf_a_node = lo;  // use cached obs_leaf
                    leaf_b_node = lo;
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

                if (lm[leaf_a_node] != LEAF_EMPTY) { fhat_a_sum[i] += lm[leaf_a_node]; fhat_a_cnt[i]++; }
                if (lm[leaf_b_node] != LEAF_EMPTY) { fhat_b_sum[i] += lm[leaf_b_node]; fhat_b_cnt[i]++; }
                if (leaf_a_node != leaf_b_node) n_diff_leaves++;
            }
        }
        // Non-splitting trees: both counterfactuals equal obs_leaf.
        // fhat_a == fhat_b == fhat_obs for these trees.
        // Since we compute PC = fa - fb, these contribute 0 to PC.
        // For the correction, we need fo = fhat_obs which is precomputed.
        // We still need to include non-splitting trees in the fhat_a/b averages
        // so that fa and fb reflect the full forest, not just splitting trees.
        else {
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1;
                int leaf = obs_leaf_flat[leaf_off + j];
                if (lm[leaf] != LEAF_EMPTY) {
                    double v = lm[leaf];
                    fhat_a_sum[i] += v; fhat_a_cnt[i]++;
                    fhat_b_sum[i] += v; fhat_b_cnt[i]++;
                }
            }
        }
    }

    // Compute phi scores
    const double* y_ptr = REAL(y_honest);
    NumericVector phi(n_hon);
    double psi_sum = 0.0; int psi_cnt = 0;
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

    double sum_pc = 0.0, sum_co = 0.0;
    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;
        double fa = (fhat_a_cnt[i] > 0) ? fhat_a_sum[i] / fhat_a_cnt[i] : NA_REAL;
        double fb = (fhat_b_cnt[i] > 0) ? fhat_b_sum[i] / fhat_b_cnt[i] : NA_REAL;
        double fo = fhat_obs_cached[j];  // precomputed in cache
        if (ISNA(fa) || ISNA(fb) || ISNA(fo)) { phi[j] = NA_REAL; continue; }
        double pc = fa - fb;
        double res = y_ptr[i] - fo;
        double xj = has_indicator ? ind_ptr[i] : X_ptr[j * p + var_col];
        double gi = g_ptr[i], w, co;
        if (is_binary) {
            double gc = std::max(0.025, std::min(0.975, gi));
            w = (xj - gc) / (gc * (1.0 - gc));
            co = w * res;
        } else {
            double ej = xj - gi;
            w = ej / sigma2_ej;
            co = w * res * (a - b);
        }
        phi[j] = pc + co;
        sum_pc += pc; sum_co += co;
        psi_sum += phi[j]; psi_cnt++;
    }

    double psi = (psi_cnt > 0) ? psi_sum / psi_cnt : NA_REAL;
    return List::create(
        Named("psi") = psi, Named("phi") = phi,
        Named("mean_pred_contrast") = (psi_cnt > 0) ? sum_pc / psi_cnt : NA_REAL,
        Named("mean_correction") = (psi_cnt > 0) ? sum_co / psi_cnt : NA_REAL,
        Named("n_contributing") = psi_cnt,
        Named("n_split_trees") = n_split_trees,
        Named("n_diff_leaves") = n_diff_leaves,
        Named("n_trees") = B);
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
        int leaf_off = bb * n_hon;

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
    double sigma2_override = -1.0
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
    IntegerVector splits_on = as<IntegerVector>(cache["splits_on"]);
    NumericVector fhat_obs_cached = as<NumericVector>(cache["fhat_obs"]);
    const double* y_ptr = REAL(y_honest);
    const double* g_ptr = REAL(ghat);
    const double* grid_ptr = REAL(grid_points);

    int G_plus_1 = grid_points.size();
    int G = G_plus_1 - 1;

    // Per-grid-point, per-observation accumulators
    std::vector<std::vector<double>> fhat_grid_sum(G_plus_1, std::vector<double>(n, 0.0));
    std::vector<std::vector<int>> fhat_grid_cnt(G_plus_1, std::vector<int>(n, 0));

    int n_split_trees = 0;

    for (int bb = 0; bb < B; bb++) {
        int off = tree_offsets[bb];
        const int* sv = INTEGER(flat_sv) + off;
        const double* sval = REAL(flat_sval) + off;
        const int* lc = INTEGER(flat_lc) + off;
        const int* rc = INTEGER(flat_rc) + off;
        const double* lm = REAL(leaf_mean_flat) + off;
        int leaf_off = bb * n_hon;

        bool tree_has_split = (splits_on[var_col * B + bb] != 0);
        if (tree_has_split) n_split_trees++;

        if (tree_has_split) {
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1;
                int xbase = j * p;

                // Walk each grid point counterfactual using shared-prefix
                // For curves, we walk G+1 grid points. Use shared prefix to
                // find the first var_col split, then branch per grid point.
                // But grid points may go different directions at each split,
                // so we need full walks per grid point (same as original).
                for (int g = 0; g < G_plus_1; g++) {
                    double gval = grid_ptr[g];
                    int node = 0;
                    while (lc[node] != 0 || rc[node] != 0) {
                        double xv = (sv[node] == var_col) ? gval : X_ptr[xbase + sv[node]];
                        node = (xv <= sval[node]) ? lc[node] : rc[node];
                    }
                    if (lm[node] != LEAF_EMPTY) {
                        fhat_grid_sum[g][i] += lm[node];
                        fhat_grid_cnt[g][i]++;
                    }
                }
            }
        } else {
            // Non-splitting tree: all grid points route to same leaf as obs
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1;
                int leaf = obs_leaf_flat[leaf_off + j];
                if (lm[leaf] != LEAF_EMPTY) {
                    double v = lm[leaf];
                    for (int g = 0; g < G_plus_1; g++) {
                        fhat_grid_sum[g][i] += v;
                        fhat_grid_cnt[g][i]++;
                    }
                }
            }
        }
    }

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

    // Compute slopes
    NumericVector slopes(G), pred_slopes(G);
    for (int g = 0; g < G; g++) {
        double dg = grid_ptr[g + 1] - grid_ptr[g];
        double phi_sum = 0.0, pred_sum = 0.0; int cnt = 0;
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double fh = (fhat_grid_cnt[g + 1][i] > 0) ? fhat_grid_sum[g + 1][i] / fhat_grid_cnt[g + 1][i] : NA_REAL;
            double fl = (fhat_grid_cnt[g][i] > 0) ? fhat_grid_sum[g][i] / fhat_grid_cnt[g][i] : NA_REAL;
            double fo = fhat_obs_cached[j];  // precomputed
            if (ISNA(fh) || ISNA(fl) || ISNA(fo)) continue;
            double pc = fh - fl, res = y_ptr[i] - fo;
            double ej = X_ptr[j * p + var_col] - g_ptr[i];
            double pg = pc + (ej / sigma2_ej) * res * dg;
            phi_sum += pg; pred_sum += pc; cnt++;
        }
        slopes[g] = (cnt > 0) ? (phi_sum / cnt) / dg : NA_REAL;
        pred_slopes[g] = (cnt > 0) ? (pred_sum / cnt) / dg : NA_REAL;
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
        Named("n_split_trees") = n_split_trees, Named("n_trees") = B);
}
