// design_variance.cpp — Analytic design-point variance (term III) for unconditional PASR
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



// ============================================================
// Phase 1: Precompute forest cache
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
