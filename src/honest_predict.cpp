// honest_predict.cpp — Honest leaf prediction for inference forests
//
// Optimizations applied:
//   - Honest routing iterates over honest_idx directly (not 0..n with skip)
//   - obs_leaf cache: leaf assignment from honest routing reused for fhat_obs
//   - Flat vector leaf means: std::vector indexed by node ID replaces
//     unordered_map. Eliminates hash lookups in the hot inner loop.
//   - Non-splitting tree skip (scores + curve): trees without var_col splits
//     contribute fhat_a == fhat_b == fhat_obs (contrast = 0, AIPW does the work)
//   - Value-override routing (aipw_scores_v2_cpp, aipw_curve_v2_cpp):
//     counterfactual routing overrides sv[node]==var_col with query value,
//     eliminating counterfactual matrix copies

#include <Rcpp.h>
#include <vector>
#include <cmath>
using namespace Rcpp;

static const double LEAF_EMPTY = -1e308;  // sentinel for empty leaves

// ============================================================
// honest_predict_cpp
// ============================================================

// [[Rcpp::export]]
NumericVector honest_predict_cpp(
    List forest, NumericMatrix X_query, NumericMatrix X_honest,
    NumericVector y_honest, IntegerVector honest_idx
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_honest.nrow();
    int n_query = X_query.nrow();
    int n_hon = honest_idx.size();
    const double* Xq_ptr = REAL(X_query);
    const double* Xh_ptr = REAL(X_honest);
    const double* y_ptr = REAL(y_honest);

    std::vector<std::vector<int>> all_sv(B), all_lc(B), all_rc(B);
    std::vector<std::vector<double>> all_sval(B);
    for (int b = 0; b < B; b++) {
        IntegerVector sv_r = svl[b]; NumericVector sval_r = svall[b];
        List ch = chl[b]; IntegerVector lc_r = ch[0], rc_r = ch[1];
        all_sv[b].assign(sv_r.begin(), sv_r.end());
        all_sval[b].assign(sval_r.begin(), sval_r.end());
        all_lc[b].assign(lc_r.begin(), lc_r.end());
        all_rc[b].assign(rc_r.begin(), rc_r.end());
    }

    std::vector<double> pred_sum(n_query, 0.0);
    std::vector<int> pred_cnt(n_query, 0);

    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();
        int n_nodes = (int)all_sv[b].size();

        std::vector<double> leaf_ysum(n_nodes, 0.0);
        std::vector<int> leaf_ycnt(n_nodes, 0);
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xh_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            leaf_ysum[node] += yi; leaf_ycnt[node]++;
        }
        std::vector<double> lm(n_nodes, LEAF_EMPTY);
        for (int nd = 0; nd < n_nodes; nd++)
            if (leaf_ycnt[nd] > 0) lm[nd] = leaf_ysum[nd] / leaf_ycnt[nd];

        for (int q = 0; q < n_query; q++) {
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xq_ptr[q + n_query * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            if (lm[node] != LEAF_EMPTY) { pred_sum[q] += lm[node]; pred_cnt[q]++; }
        }
    }

    NumericVector result(n_query);
    for (int q = 0; q < n_query; q++)
        result[q] = (pred_cnt[q] > 0) ? pred_sum[q] / pred_cnt[q] : NA_REAL;
    return result;
}

