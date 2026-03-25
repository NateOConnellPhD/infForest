// aipw_curve.cpp — AIPW curve estimation for inference forests
//
// Grid-based slope and level curve estimation with value-override routing.

#include <Rcpp.h>
#include <vector>
#include <cmath>
using namespace Rcpp;

static const double LEAF_EMPTY = -1e308;

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
    NumericMatrix phi_scores(n_hon, G);
    std::fill(phi_scores.begin(), phi_scores.end(), NA_REAL);

    // Build fhat_grid (n_hon x G+1) and fhat_obs (n_hon) for R
    NumericMatrix fhat_grid(n_hon, G_plus_1);
    NumericVector fhat_obs(n_hon);
    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;
        fhat_obs[j] = (fhat_obs_cnt[i] > 0) ? fhat_obs_sum[i] / fhat_obs_cnt[i] : NA_REAL;
        for (int g = 0; g < G_plus_1; g++) {
            fhat_grid(j, g) = (fhat_grid_cnt[g][i] > 0) ? fhat_grid_sum[g][i] / fhat_grid_cnt[g][i] : NA_REAL;
        }
    }

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
            phi_scores(j, g) = pg / dg;  // per-unit phi score
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
        Named("n_split_trees")=n_split_trees, Named("n_trees")=B,
        Named("phi_scores")=phi_scores,
        Named("fhat_grid")=fhat_grid, Named("fhat_obs")=fhat_obs);
}


// Backward-compatible wrapper
