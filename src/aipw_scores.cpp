// aipw_scores.cpp — AIPW score computation for inference forests
//
// Value-override routing: counterfactual routing overrides sv[node]==var_col
// with query value, eliminating counterfactual matrix copies.

#include <Rcpp.h>
#include <vector>
#include <cmath>
using namespace Rcpp;

static const double LEAF_EMPTY = -1e308;

// ============================================================
// aipw_scores_v2_cpp — no counterfactual matrices, value-override routing
// ============================================================

// [[Rcpp::export]]
List aipw_scores_v2_cpp(
    List forest, NumericMatrix X_obs, NumericVector y_honest,
    IntegerVector honest_idx, NumericVector ghat,
    int var_col, bool is_binary, double a, double b,
    Nullable<NumericVector> indicator_ = R_NilValue
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size(); int n = X_obs.nrow();
    const double* Xobs_ptr = REAL(X_obs);
    const double* y_ptr = REAL(y_honest);
    const double* g_ptr = REAL(ghat);
    int n_hon = honest_idx.size();

    // Optional indicator vector for HT weights (categorical predictors)
    bool has_indicator = indicator_.isNotNull();
    NumericVector indicator_vec;
    const double* ind_ptr = nullptr;
    if (has_indicator) {
        indicator_vec = as<NumericVector>(indicator_);
        ind_ptr = REAL(indicator_vec);
    }

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
        } else {
            // NON-SPLITTING TREE (binary or continuous): all routes identical.
            // Prediction contrast = 0; AIPW correction does the work.
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
    NumericVector fhat_a_out(n_hon), fhat_b_out(n_hon), fhat_obs_out(n_hon);
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
        fhat_a_out[j] = fa; fhat_b_out[j] = fb; fhat_obs_out[j] = fo;
        if (ISNA(fa)||ISNA(fb)||ISNA(fo)) { phi[j]=NA_REAL; continue; }
        double pc = fa - fb, res = y_ptr[i] - fo;
        double xj = has_indicator ? ind_ptr[i] : Xobs_ptr[i+n*var_col]; double gi = g_ptr[i], w, co;
        if (is_binary) { double gc = std::max(0.025,std::min(0.975,gi)); w = (xj-gc)/(gc*(1.0-gc)); co = w*res; }
        else { double ej = xj-gi; w = ej/sigma2_ej; co = w*res*(a-b); }
        phi[j] = pc + co; sum_pc += pc; sum_co += co; psi_sum += phi[j]; psi_cnt++;
    }
    double psi = (psi_cnt>0) ? psi_sum/psi_cnt : NA_REAL;
    return List::create(Named("psi")=psi, Named("phi")=phi,
        Named("mean_pred_contrast")=(psi_cnt>0)?sum_pc/psi_cnt:NA_REAL,
        Named("mean_correction")=(psi_cnt>0)?sum_co/psi_cnt:NA_REAL,
        Named("n_contributing")=psi_cnt, Named("n_split_trees")=n_split_trees, Named("n_trees")=B,
        Named("fhat_a")=fhat_a_out, Named("fhat_b")=fhat_b_out, Named("fhat_obs")=fhat_obs_out,
        Named("sigma2_ej")=sigma2_ej);
}

