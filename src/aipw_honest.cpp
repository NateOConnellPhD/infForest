// aipw_honest.cpp — AIPW effect estimation for inference forests
//
// Optimizations applied:
//   - Honest routing iterates over honest_idx directly (not 0..n with skip)
//   - obs_leaf cache: leaf assignment from honest routing reused for fhat_obs
//   - Flat vector leaf means: std::vector indexed by node ID replaces
//     unordered_map. Eliminates hash lookups in the hot inner loop.
//   - Non-splitting tree skip (scores + curve)
//   - Fused augmentation for binary (aipw_scores_v2_cpp)
//   - Grid vector routing (aipw_curve_v2_cpp)

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


// ============================================================
// honest_predict_loo_cpp
// ============================================================

// [[Rcpp::export]]
NumericVector honest_predict_loo_cpp(
    List forest, NumericMatrix X_honest,
    NumericVector y_honest, IntegerVector honest_idx
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_honest.nrow();
    int n_hon = honest_idx.size();
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

    std::vector<double> pred_sum(n, 0.0);
    std::vector<int> pred_cnt(n, 0);

    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();
        int n_nodes = (int)all_sv[b].size();

        std::vector<double> leaf_ysum(n_nodes, 0.0);
        std::vector<int> leaf_ycnt(n_nodes, 0);
        std::vector<int> obs_leaf(n_hon, -1);
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xh_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            obs_leaf[j] = node;
            leaf_ysum[node] += yi; leaf_ycnt[node]++;
        }

        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            int leaf = obs_leaf[j];
            if (leaf < 0) continue;
            int cnt = leaf_ycnt[leaf];
            if (cnt <= 1) continue;
            pred_sum[i] += (leaf_ysum[leaf] - yi) / (cnt - 1);
            pred_cnt[i]++;
        }
    }

    NumericVector result(n, NA_REAL);
    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;
        if (pred_cnt[i] > 0) result[i] = pred_sum[i] / pred_cnt[i];
    }
    return result;
}


// ============================================================
// aipw_scores_cpp (backward-compatible)
// ============================================================

// [[Rcpp::export]]
List aipw_scores_cpp(
    List forest, NumericMatrix X_obs,
    NumericMatrix X_query_a, NumericMatrix X_query_b,
    NumericVector y_honest, IntegerVector honest_idx, NumericVector ghat,
    int var_col, bool is_binary, double a, double b
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size(); int n = X_obs.nrow();
    const double* Xobs_ptr = REAL(X_obs);
    const double* Xa_ptr = REAL(X_query_a);
    const double* Xb_ptr = REAL(X_query_b);
    const double* y_ptr = REAL(y_honest);
    const double* g_ptr = REAL(ghat);
    int n_hon = honest_idx.size();

    std::vector<std::vector<int>> all_sv(B), all_lc(B), all_rc(B);
    std::vector<std::vector<double>> all_sval(B);
    for (int bb = 0; bb < B; bb++) {
        IntegerVector sv_r = svl[bb]; NumericVector sval_r = svall[bb];
        List ch = chl[bb]; IntegerVector lc_r = ch[0], rc_r = ch[1];
        all_sv[bb].assign(sv_r.begin(), sv_r.end());
        all_sval[bb].assign(sval_r.begin(), sval_r.end());
        all_lc[bb].assign(lc_r.begin(), lc_r.end());
        all_rc[bb].assign(rc_r.begin(), rc_r.end());
    }

    std::vector<double> fhat_a_sum(n, 0.0), fhat_b_sum(n, 0.0), fhat_obs_sum(n, 0.0);
    std::vector<int> fhat_a_cnt(n, 0), fhat_b_cnt(n, 0), fhat_obs_cnt(n, 0);

    int n_split_trees = 0;
    std::vector<bool> tree_splits(B, false);
    for (int bb = 0; bb < B; bb++)
        for (int nd = 0; nd < (int)all_sv[bb].size(); nd++)
            if (all_lc[bb][nd] != 0 && all_sv[bb][nd] == var_col) {
                tree_splits[bb] = true; n_split_trees++; break;
            }

    for (int bb = 0; bb < B; bb++) {
        const int* sv = all_sv[bb].data(); const double* sval = all_sval[bb].data();
        const int* lc = all_lc[bb].data(); const int* rc = all_rc[bb].data();
        int n_nodes = (int)all_sv[bb].size();

        std::vector<double> leaf_ysum(n_nodes, 0.0);
        std::vector<int> leaf_ycnt(n_nodes, 0);
        std::vector<int> obs_leaf(n_hon);
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1; double yi = y_ptr[i];
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xobs_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            obs_leaf[j] = node;
            if (!ISNA(yi)) { leaf_ysum[node] += yi; leaf_ycnt[node]++; }
        }
        std::vector<double> lm(n_nodes, LEAF_EMPTY);
        for (int nd = 0; nd < n_nodes; nd++)
            if (leaf_ycnt[nd] > 0) lm[nd] = leaf_ysum[nd] / leaf_ycnt[nd];

        if (tree_splits[bb]) {
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1;
                int lo = obs_leaf[j];
                if (lm[lo] != LEAF_EMPTY) { fhat_obs_sum[i] += lm[lo]; fhat_obs_cnt[i]++; }
                { int node = 0;
                  while (lc[node] != 0 || rc[node] != 0)
                      node = (Xa_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
                  if (lm[node] != LEAF_EMPTY) { fhat_a_sum[i] += lm[node]; fhat_a_cnt[i]++; } }
                { int node = 0;
                  while (lc[node] != 0 || rc[node] != 0)
                      node = (Xb_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
                  if (lm[node] != LEAF_EMPTY) { fhat_b_sum[i] += lm[node]; fhat_b_cnt[i]++; } }
            }
        } else {
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1; int leaf = obs_leaf[j];
                if (lm[leaf] != LEAF_EMPTY) {
                    double v = lm[leaf];
                    fhat_obs_sum[i] += v; fhat_obs_cnt[i]++;
                    fhat_a_sum[i] += v; fhat_a_cnt[i]++;
                    fhat_b_sum[i] += v; fhat_b_cnt[i]++;
                }
            }
        }
    }

    NumericVector phi(n_hon);
    double psi_sum = 0.0; int psi_cnt = 0;
    double sigma2_ej = 0.0;
    if (!is_binary) {
        double ss = 0.0;
        for (int j = 0; j < n_hon; j++) { int i = honest_idx[j]-1; double ej = Xobs_ptr[i+n*var_col]-g_ptr[i]; ss += ej*ej; }
        sigma2_ej = ss / n_hon;
    }
    double sum_pc = 0.0, sum_co = 0.0;
    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;
        double fa = (fhat_a_cnt[i]>0) ? fhat_a_sum[i]/fhat_a_cnt[i] : NA_REAL;
        double fb = (fhat_b_cnt[i]>0) ? fhat_b_sum[i]/fhat_b_cnt[i] : NA_REAL;
        double fo = (fhat_obs_cnt[i]>0) ? fhat_obs_sum[i]/fhat_obs_cnt[i] : NA_REAL;
        if (ISNA(fa)||ISNA(fb)||ISNA(fo)) { phi[j]=NA_REAL; continue; }
        double pc = fa - fb, res = y_ptr[i] - fo;
        double xj = Xobs_ptr[i+n*var_col], gi = g_ptr[i], w, co;
        if (is_binary) { double gc = std::max(0.025,std::min(0.975,gi)); w = (xj-gc)/(gc*(1.0-gc)); co = w*res; }
        else { double ej = xj-gi; w = ej/sigma2_ej; co = w*res*(a-b); }
        phi[j] = pc + co; sum_pc += pc; sum_co += co; psi_sum += phi[j]; psi_cnt++;
    }
    double psi = (psi_cnt>0) ? psi_sum/psi_cnt : NA_REAL;
    return List::create(Named("psi")=psi, Named("phi")=phi,
        Named("mean_pred_contrast")=(psi_cnt>0)?sum_pc/psi_cnt:NA_REAL,
        Named("mean_correction")=(psi_cnt>0)?sum_co/psi_cnt:NA_REAL,
        Named("n_contributing")=psi_cnt, Named("n_split_trees")=n_split_trees, Named("n_trees")=B);
}


// ============================================================
// aipw_scores_v2_cpp — no counterfactual matrices, fused augmentation
// ============================================================

// [[Rcpp::export]]
List aipw_scores_v2_cpp(
    List forest, NumericMatrix X_obs, NumericVector y_honest,
    IntegerVector honest_idx, NumericVector ghat,
    int var_col, bool is_binary, double a, double b
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size(); int n = X_obs.nrow();
    const double* Xobs_ptr = REAL(X_obs);
    const double* y_ptr = REAL(y_honest);
    const double* g_ptr = REAL(ghat);
    int n_hon = honest_idx.size();

    std::vector<std::vector<int>> all_sv(B), all_lc(B), all_rc(B);
    std::vector<std::vector<double>> all_sval(B);
    for (int bb = 0; bb < B; bb++) {
        IntegerVector sv_r = svl[bb]; NumericVector sval_r = svall[bb];
        List ch = chl[bb]; IntegerVector lc_r = ch[0], rc_r = ch[1];
        all_sv[bb].assign(sv_r.begin(), sv_r.end());
        all_sval[bb].assign(sval_r.begin(), sval_r.end());
        all_lc[bb].assign(lc_r.begin(), lc_r.end());
        all_rc[bb].assign(rc_r.begin(), rc_r.end());
    }

    std::vector<double> fhat_a_sum(n, 0.0), fhat_b_sum(n, 0.0), fhat_obs_sum(n, 0.0);
    std::vector<int> fhat_a_cnt(n, 0), fhat_b_cnt(n, 0), fhat_obs_cnt(n, 0);

    int n_split_trees = 0;
    std::vector<bool> tree_splits(B, false);
    for (int bb = 0; bb < B; bb++)
        for (int nd = 0; nd < (int)all_sv[bb].size(); nd++)
            if (all_lc[bb][nd] != 0 && all_sv[bb][nd] == var_col) {
                tree_splits[bb] = true; n_split_trees++; break;
            }

    for (int bb = 0; bb < B; bb++) {
        const int* sv = all_sv[bb].data(); const double* sval = all_sval[bb].data();
        const int* lc = all_lc[bb].data(); const int* rc = all_rc[bb].data();
        int n_nodes = (int)all_sv[bb].size();

        std::vector<double> leaf_ysum(n_nodes, 0.0);
        std::vector<int> leaf_ycnt(n_nodes, 0);
        std::vector<int> obs_leaf(n_hon);
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1; double yi = y_ptr[i];
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xobs_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            obs_leaf[j] = node;
            if (!ISNA(yi)) { leaf_ysum[node] += yi; leaf_ycnt[node]++; }
        }
        std::vector<double> lm(n_nodes, LEAF_EMPTY);
        for (int nd = 0; nd < n_nodes; nd++)
            if (leaf_ycnt[nd] > 0) lm[nd] = leaf_ysum[nd] / leaf_ycnt[nd];

        if (tree_splits[bb]) {
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1;
                int lo = obs_leaf[j];
                if (lm[lo] != LEAF_EMPTY) { fhat_obs_sum[i] += lm[lo]; fhat_obs_cnt[i]++; }
                { int node = 0;
                  while (lc[node] != 0 || rc[node] != 0) {
                      double xv = (sv[node]==var_col) ? a : Xobs_ptr[i+n*sv[node]];
                      node = (xv <= sval[node]) ? lc[node] : rc[node]; }
                  if (lm[node] != LEAF_EMPTY) { fhat_a_sum[i] += lm[node]; fhat_a_cnt[i]++; } }
                { int node = 0;
                  while (lc[node] != 0 || rc[node] != 0) {
                      double xv = (sv[node]==var_col) ? b : Xobs_ptr[i+n*sv[node]];
                      node = (xv <= sval[node]) ? lc[node] : rc[node]; }
                  if (lm[node] != LEAF_EMPTY) { fhat_b_sum[i] += lm[node]; fhat_b_cnt[i]++; } }
            }
        } else if (is_binary) {
            // Fused inline augmentation
            // Build per-leaf X_j-group sums for daughter means
            std::vector<double> ly_hi(n_nodes, 0.0), ly_lo(n_nodes, 0.0);
            std::vector<int> nc_hi(n_nodes, 0), nc_lo(n_nodes, 0);
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1; double yi = y_ptr[i];
                if (ISNA(yi)) continue;
                int leaf = obs_leaf[j];
                if (Xobs_ptr[i + n * var_col] > 0.5) { ly_hi[leaf] += yi; nc_hi[leaf]++; }
                else                                   { ly_lo[leaf] += yi; nc_lo[leaf]++; }
            }
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1; int leaf = obs_leaf[j];
                // Min-count guard: if either daughter has <2 obs,
                // augmentation is variance-increasing. Fall back to unsplit.
                if (nc_hi[leaf] < 2 || nc_lo[leaf] < 2) {
                    if (lm[leaf] != LEAF_EMPTY) {
                        fhat_obs_sum[i] += lm[leaf]; fhat_obs_cnt[i]++;
                        fhat_a_sum[i]   += lm[leaf]; fhat_a_cnt[i]++;
                        fhat_b_sum[i]   += lm[leaf]; fhat_b_cnt[i]++;
                    }
                } else {
                    // Augmented: fhat_obs routes through the daughter matching
                    // this obs's actual X_j value (consistent with augmented tree)
                    if (Xobs_ptr[i + n * var_col] > 0.5) {
                        fhat_obs_sum[i] += ly_hi[leaf]/nc_hi[leaf]; fhat_obs_cnt[i]++;
                    } else {
                        fhat_obs_sum[i] += ly_lo[leaf]/nc_lo[leaf]; fhat_obs_cnt[i]++;
                    }
                    fhat_a_sum[i] += ly_hi[leaf]/nc_hi[leaf]; fhat_a_cnt[i]++;
                    fhat_b_sum[i] += ly_lo[leaf]/nc_lo[leaf]; fhat_b_cnt[i]++;
                }
            }
        } else {
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1; int leaf = obs_leaf[j];
                if (lm[leaf] != LEAF_EMPTY) {
                    double v = lm[leaf];
                    fhat_obs_sum[i] += v; fhat_obs_cnt[i]++;
                    fhat_a_sum[i] += v; fhat_a_cnt[i]++;
                    fhat_b_sum[i] += v; fhat_b_cnt[i]++;
                }
            }
        }
    }

    NumericVector phi(n_hon);
    double psi_sum = 0.0; int psi_cnt = 0;
    double sigma2_ej = 0.0;
    if (!is_binary) {
        double ss = 0.0;
        for (int j = 0; j < n_hon; j++) { int i = honest_idx[j]-1; double ej = Xobs_ptr[i+n*var_col]-g_ptr[i]; ss += ej*ej; }
        sigma2_ej = ss / n_hon;
    }
    double sum_pc = 0.0, sum_co = 0.0;
    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;
        double fa = (fhat_a_cnt[i]>0) ? fhat_a_sum[i]/fhat_a_cnt[i] : NA_REAL;
        double fb = (fhat_b_cnt[i]>0) ? fhat_b_sum[i]/fhat_b_cnt[i] : NA_REAL;
        double fo = (fhat_obs_cnt[i]>0) ? fhat_obs_sum[i]/fhat_obs_cnt[i] : NA_REAL;
        if (ISNA(fa)||ISNA(fb)||ISNA(fo)) { phi[j]=NA_REAL; continue; }
        double pc = fa - fb, res = y_ptr[i] - fo;
        double xj = Xobs_ptr[i+n*var_col], gi = g_ptr[i], w, co;
        if (is_binary) { double gc = std::max(0.025,std::min(0.975,gi)); w = (xj-gc)/(gc*(1.0-gc)); co = w*res; }
        else { double ej = xj-gi; w = ej/sigma2_ej; co = w*res*(a-b); }
        phi[j] = pc + co; sum_pc += pc; sum_co += co; psi_sum += phi[j]; psi_cnt++;
    }
    double psi = (psi_cnt>0) ? psi_sum/psi_cnt : NA_REAL;
    return List::create(Named("psi")=psi, Named("phi")=phi,
        Named("mean_pred_contrast")=(psi_cnt>0)?sum_pc/psi_cnt:NA_REAL,
        Named("mean_correction")=(psi_cnt>0)?sum_co/psi_cnt:NA_REAL,
        Named("n_contributing")=psi_cnt, Named("n_split_trees")=n_split_trees, Named("n_trees")=B);
}


// ============================================================
// aipw_curve_v2_cpp — grid vector, no matrix list
// ============================================================

// [[Rcpp::export]]
List aipw_curve_v2_cpp(
    List forest, NumericMatrix X_obs, NumericVector y_honest,
    IntegerVector honest_idx, NumericVector ghat,
    int var_col, NumericVector grid_points, double sigma2_override = -1.0
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size(); int n = X_obs.nrow();
    int n_hon = honest_idx.size();
    int G_plus_1 = grid_points.size(); int G = G_plus_1 - 1;
    const double* Xobs_ptr = REAL(X_obs);
    const double* y_ptr = REAL(y_honest);
    const double* g_ptr = REAL(ghat);
    const double* grid_ptr = REAL(grid_points);

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

    std::vector<double> fhat_obs_sum(n, 0.0);
    std::vector<int> fhat_obs_cnt(n, 0);
    std::vector<std::vector<double>> fhat_grid_sum(G_plus_1, std::vector<double>(n, 0.0));
    std::vector<std::vector<int>> fhat_grid_cnt(G_plus_1, std::vector<int>(n, 0));

    int n_split_trees = 0;
    std::vector<bool> tree_splits(B, false);
    for (int b = 0; b < B; b++)
        for (int nd = 0; nd < (int)all_sv[b].size(); nd++)
            if (all_lc[b][nd] != 0 && all_sv[b][nd] == var_col) {
                tree_splits[b] = true; n_split_trees++; break;
            }

    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data(); const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data(); const int* rc = all_rc[b].data();
        int n_nodes = (int)all_sv[b].size();

        std::vector<double> leaf_ysum(n_nodes, 0.0);
        std::vector<int> leaf_ycnt(n_nodes, 0);
        std::vector<int> obs_leaf(n_hon);
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1; double yi = y_ptr[i];
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xobs_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            obs_leaf[j] = node;
            if (!ISNA(yi)) { leaf_ysum[node] += yi; leaf_ycnt[node]++; }
        }
        std::vector<double> lm(n_nodes, LEAF_EMPTY);
        for (int nd = 0; nd < n_nodes; nd++)
            if (leaf_ycnt[nd] > 0) lm[nd] = leaf_ysum[nd] / leaf_ycnt[nd];

        if (tree_splits[b]) {
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1;
                int lo = obs_leaf[j];
                if (lm[lo] != LEAF_EMPTY) { fhat_obs_sum[i] += lm[lo]; fhat_obs_cnt[i]++; }
                for (int g = 0; g < G_plus_1; g++) {
                    double gval = grid_ptr[g]; int node = 0;
                    while (lc[node] != 0 || rc[node] != 0) {
                        double xv = (sv[node]==var_col) ? gval : Xobs_ptr[i+n*sv[node]];
                        node = (xv <= sval[node]) ? lc[node] : rc[node];
                    }
                    if (lm[node] != LEAF_EMPTY) { fhat_grid_sum[g][i] += lm[node]; fhat_grid_cnt[g][i]++; }
                }
            }
        } else {
            for (int j = 0; j < n_hon; j++) {
                int i = honest_idx[j] - 1; int leaf = obs_leaf[j];
                if (lm[leaf] != LEAF_EMPTY) {
                    double v = lm[leaf];
                    fhat_obs_sum[i] += v; fhat_obs_cnt[i]++;
                    for (int g = 0; g < G_plus_1; g++) {
                        fhat_grid_sum[g][i] += v; fhat_grid_cnt[g][i]++;
                    }
                }
            }
        }
    }

    double sigma2_ej;
    if (sigma2_override > 0.0) { sigma2_ej = sigma2_override; }
    else {
        double ss = 0.0;
        for (int j = 0; j < n_hon; j++) { int i = honest_idx[j]-1; double ej = Xobs_ptr[i+n*var_col]-g_ptr[i]; ss += ej*ej; }
        sigma2_ej = ss / n_hon;
    }

    NumericVector slopes(G), pred_slopes(G);
    for (int g = 0; g < G; g++) {
        double dg = grid_points[g+1] - grid_points[g];
        double phi_sum = 0.0, pred_sum = 0.0; int cnt = 0;
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double fh = (fhat_grid_cnt[g+1][i]>0) ? fhat_grid_sum[g+1][i]/fhat_grid_cnt[g+1][i] : NA_REAL;
            double fl = (fhat_grid_cnt[g][i]>0) ? fhat_grid_sum[g][i]/fhat_grid_cnt[g][i] : NA_REAL;
            double fo = (fhat_obs_cnt[i]>0) ? fhat_obs_sum[i]/fhat_obs_cnt[i] : NA_REAL;
            if (ISNA(fh)||ISNA(fl)||ISNA(fo)) continue;
            double pc = fh - fl, res = y_ptr[i] - fo;
            double ej = Xobs_ptr[i+n*var_col] - g_ptr[i];
            double pg = pc + (ej/sigma2_ej)*res*dg;
            phi_sum += pg; pred_sum += pc; cnt++;
        }
        slopes[g] = (cnt>0) ? (phi_sum/cnt)/dg : NA_REAL;
        pred_slopes[g] = (cnt>0) ? (pred_sum/cnt)/dg : NA_REAL;
    }

    NumericVector curve(G_plus_1, 0.0);
    for (int g = 0; g < G; g++) {
        double dg = grid_points[g+1] - grid_points[g];
        curve[g+1] = curve[g] + (ISNA(slopes[g]) ? 0.0 : slopes[g]*dg);
    }

    return List::create(
        Named("slopes")=slopes, Named("pred_slopes")=pred_slopes,
        Named("curve")=curve, Named("grid")=grid_points,
        Named("sigma2_ej")=sigma2_ej, Named("n_hon")=n_hon,
        Named("n_split_trees")=n_split_trees, Named("n_trees")=B);
}


// Backward-compatible wrapper
// [[Rcpp::export]]
List aipw_curve_cpp(
    List forest, NumericMatrix X_obs, List X_grid_list,
    NumericVector y_honest, IntegerVector honest_idx, NumericVector ghat,
    int var_col, NumericVector grid_points, double sigma2_override = -1.0
) {
    return aipw_curve_v2_cpp(forest, X_obs, y_honest, honest_idx,
                              ghat, var_col, grid_points, sigma2_override);
}
