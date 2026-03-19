// aipw_honest.cpp — AIPW effect estimation for inference forests
//
// All trees contribute to all three prediction sets (fhat_a, fhat_b, fhat_obs).
// Honest leaf means from Y_B. Residual = Y_k^B - fhat_obs(k).
// Propensity weights from R (OOB predictions).
//
// For each tree:
//   1. Route honest obs to leaves, compute leaf means (ONCE)
//   2. Route fhat_obs, fhat_a, fhat_b query sets, look up means

#include <Rcpp.h>
#include <vector>
#include <unordered_map>
using namespace Rcpp;


// Standalone honest prediction at arbitrary query points
// [[Rcpp::export]]
NumericVector honest_predict_cpp(
    List forest,
    NumericMatrix X_query,
    NumericMatrix X_honest,
    NumericVector y_honest,
    IntegerVector honest_idx
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_honest.nrow();
    int n_query = X_query.nrow();

    const double* Xq_ptr = REAL(X_query);
    const double* Xh_ptr = REAL(X_honest);
    const double* y_ptr = REAL(y_honest);

    std::vector<bool> is_honest(n, false);
    for (int j = 0; j < honest_idx.size(); j++)
        is_honest[honest_idx[j] - 1] = true;

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

        std::unordered_map<int, double> leaf_ysum;
        std::unordered_map<int, int> leaf_ycnt;
        for (int i = 0; i < n; i++) {
            if (!is_honest[i]) continue;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xh_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            leaf_ysum[node] += yi;
            leaf_ycnt[node]++;
        }
        std::unordered_map<int, double> leaf_mean;
        for (auto& kv : leaf_ycnt)
            if (kv.second > 0)
                leaf_mean[kv.first] = leaf_ysum[kv.first] / kv.second;

        for (int q = 0; q < n_query; q++) {
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xq_ptr[q + n_query * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            auto it = leaf_mean.find(node);
            if (it != leaf_mean.end()) {
                pred_sum[q] += it->second;
                pred_cnt[q]++;
            }
        }
    }

    NumericVector result(n_query);
    for (int q = 0; q < n_query; q++)
        result[q] = (pred_cnt[q] > 0) ? pred_sum[q] / pred_cnt[q] : NA_REAL;
    return result;
}


// Leave-one-out honest prediction at honest observation positions.
//
// For each honest observation k, predicts f_{-k}(X_k): the honest leaf mean
// in each tree EXCLUDING Y_k from the leaf sum. This ensures the residual
// Y_k - f_{-k}(X_k) is independent of the leaf mean, preventing the
// propensity-weighted residual from having a systematic nonzero mean.
//
// Returns a length-n vector (NA for non-honest observations).
//
// [[Rcpp::export]]
NumericVector honest_predict_loo_cpp(
    List forest,
    NumericMatrix X_honest,
    NumericVector y_honest,
    IntegerVector honest_idx
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_honest.nrow();
    int n_hon = honest_idx.size();

    const double* Xh_ptr = REAL(X_honest);
    const double* y_ptr = REAL(y_honest);

    std::vector<bool> is_honest(n, false);
    for (int j = 0; j < n_hon; j++)
        is_honest[honest_idx[j] - 1] = true;

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

    // Accumulators for each honest observation: sum of LOO predictions, count of trees
    std::vector<double> pred_sum(n, 0.0);
    std::vector<int> pred_cnt(n, 0);

    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();

        // Route all honest obs to leaves, compute leaf sums and counts
        std::unordered_map<int, double> leaf_ysum;
        std::unordered_map<int, int> leaf_ycnt;
        std::vector<int> obs_leaf(n, -1);

        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xh_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            obs_leaf[i] = node;
            leaf_ysum[node] += yi;
            leaf_ycnt[node]++;
        }

        // For each honest obs: LOO prediction = (leaf_sum - Y_k) / (leaf_count - 1)
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            int leaf = obs_leaf[i];
            if (leaf < 0) continue;

            auto it_sum = leaf_ysum.find(leaf);
            auto it_cnt = leaf_ycnt.find(leaf);
            if (it_sum == leaf_ysum.end() || it_cnt == leaf_ycnt.end()) continue;

            int cnt = it_cnt->second;
            if (cnt <= 1) continue;  // can't do LOO with only 1 obs in leaf

            double loo_mean = (it_sum->second - yi) / (cnt - 1);
            pred_sum[i] += loo_mean;
            pred_cnt[i]++;
        }
    }

    NumericVector result(n, NA_REAL);
    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;
        if (pred_cnt[i] > 0)
            result[i] = pred_sum[i] / pred_cnt[i];
    }
    return result;
}


// AIPW scores for one fold direction, one variable.
//
// All trees contribute to fhat_obs, fhat_a, fhat_b.
// Honest leaf means for all three. Residual = Y_k - fhat_obs(k).
//
// [[Rcpp::export]]
List aipw_scores_cpp(
    List forest,
    NumericMatrix X_obs,       // n x p: actual covariates
    NumericMatrix X_query_a,   // n x p: counterfactual with X_j = a
    NumericMatrix X_query_b,   // n x p: counterfactual with X_j = b
    NumericVector y_honest,    // length n: honest outcomes (NA for non-honest)
    IntegerVector honest_idx,  // 1-indexed honest observation indices
    NumericVector ghat,        // length n: propensity predictions (OOB)
    int var_col,               // 0-indexed column of variable of interest
    bool is_binary,
    double a,
    double b
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_obs.nrow();

    const double* Xobs_ptr = REAL(X_obs);
    const double* Xa_ptr = REAL(X_query_a);
    const double* Xb_ptr = REAL(X_query_b);
    const double* y_ptr = REAL(y_honest);
    const double* g_ptr = REAL(ghat);

    int n_hon = honest_idx.size();

    std::vector<bool> is_honest(n, false);
    for (int j = 0; j < n_hon; j++)
        is_honest[honest_idx[j] - 1] = true;

    // Pre-extract forest
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

    // Accumulators for all three prediction sets
    std::vector<double> fhat_a_sum(n, 0.0), fhat_b_sum(n, 0.0), fhat_obs_sum(n, 0.0);
    std::vector<int> fhat_a_cnt(n, 0), fhat_b_cnt(n, 0), fhat_obs_cnt(n, 0);

    // Diagnostic: count trees that split on var_col
    int n_split_trees = 0;
    for (int b = 0; b < B; b++) {
        for (int nd = 0; nd < (int)all_sv[b].size(); nd++) {
            if (all_lc[b][nd] != 0 && all_sv[b][nd] == var_col) {
                n_split_trees++;
                break;
            }
        }
    }

    // ========== TREE LOOP ==========
    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();

        // Route honest obs to leaves, compute honest leaf means (ONCE)
        std::unordered_map<int, double> leaf_ysum;
        std::unordered_map<int, int> leaf_ycnt;
        for (int i = 0; i < n; i++) {
            if (!is_honest[i]) continue;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xobs_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            leaf_ysum[node] += yi;
            leaf_ycnt[node]++;
        }
        std::unordered_map<int, double> leaf_mean;
        for (auto& kv : leaf_ycnt)
            if (kv.second > 0)
                leaf_mean[kv.first] = leaf_ysum[kv.first] / kv.second;

        // Route all three query sets for ALL trees
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;

            // fhat_obs
            {
                int node = 0;
                while (lc[node] != 0 || rc[node] != 0)
                    node = (Xobs_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
                auto it = leaf_mean.find(node);
                if (it != leaf_mean.end()) { fhat_obs_sum[i] += it->second; fhat_obs_cnt[i]++; }
            }

            // fhat_a
            {
                int node = 0;
                while (lc[node] != 0 || rc[node] != 0)
                    node = (Xa_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
                auto it = leaf_mean.find(node);
                if (it != leaf_mean.end()) { fhat_a_sum[i] += it->second; fhat_a_cnt[i]++; }
            }

            // fhat_b
            {
                int node = 0;
                while (lc[node] != 0 || rc[node] != 0)
                    node = (Xb_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
                auto it = leaf_mean.find(node);
                if (it != leaf_mean.end()) { fhat_b_sum[i] += it->second; fhat_b_cnt[i]++; }
            }
        }
    }
    // ========== END TREE LOOP ==========

    // Compute AIPW scores
    NumericVector phi(n_hon);
    double psi_sum = 0.0;
    int psi_cnt = 0;

    double sigma2_ej = 0.0;
    if (!is_binary) {
        double ss = 0.0;
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double ej = Xobs_ptr[i + n * var_col] - g_ptr[i];
            ss += ej * ej;
        }
        sigma2_ej = ss / n_hon;
    }

    double sum_pred_contrast = 0.0;
    double sum_correction = 0.0;

    for (int j = 0; j < n_hon; j++) {
        int i = honest_idx[j] - 1;

        double fa = (fhat_a_cnt[i] > 0) ? fhat_a_sum[i] / fhat_a_cnt[i] : NA_REAL;
        double fb = (fhat_b_cnt[i] > 0) ? fhat_b_sum[i] / fhat_b_cnt[i] : NA_REAL;
        double fo = (fhat_obs_cnt[i] > 0) ? fhat_obs_sum[i] / fhat_obs_cnt[i] : NA_REAL;

        if (ISNA(fa) || ISNA(fb) || ISNA(fo)) {
            phi[j] = NA_REAL;
            continue;
        }

        double pred_contrast = fa - fb;
        double residual = y_ptr[i] - fo;
        double xj = Xobs_ptr[i + n * var_col];
        double gi = g_ptr[i];

        double weight, correction;
        if (is_binary) {
            double gi_clip = std::max(0.025, std::min(0.975, gi));
            weight = (xj - gi_clip) / (gi_clip * (1.0 - gi_clip));
            correction = weight * residual;
        } else {
            double ej = xj - gi;
            weight = ej / sigma2_ej;
            correction = weight * residual * (a - b);
        }

        phi[j] = pred_contrast + correction;
        sum_pred_contrast += pred_contrast;
        sum_correction += correction;
        psi_sum += phi[j];
        psi_cnt++;
    }

    double psi = (psi_cnt > 0) ? psi_sum / psi_cnt : NA_REAL;

    return List::create(
        Named("psi") = psi,
        Named("phi") = phi,
        Named("mean_pred_contrast") = (psi_cnt > 0) ? sum_pred_contrast / psi_cnt : NA_REAL,
        Named("mean_correction") = (psi_cnt > 0) ? sum_correction / psi_cnt : NA_REAL,
        Named("n_contributing") = psi_cnt,
        Named("n_split_trees") = n_split_trees,
        Named("n_trees") = B
    );
}


// AIPW effect curves: predict at G+1 grid points + fhat_obs in a single tree loop.
//
// [[Rcpp::export]]
List aipw_curve_cpp(
    List forest,
    NumericMatrix X_obs,
    List X_grid_list,          // List of G+1 matrices (n x p)
    NumericVector y_honest,
    IntegerVector honest_idx,
    NumericVector ghat,
    int var_col,
    NumericVector grid_points, // G+1 grid values
    double sigma2_override = -1.0  // if positive, use instead of internal sigma2_ej
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_obs.nrow();
    int n_hon = honest_idx.size();
    int G_plus_1 = X_grid_list.size();
    int G = G_plus_1 - 1;

    const double* Xobs_ptr = REAL(X_obs);
    const double* y_ptr = REAL(y_honest);
    const double* g_ptr = REAL(ghat);

    std::vector<bool> is_honest(n, false);
    for (int j = 0; j < n_hon; j++)
        is_honest[honest_idx[j] - 1] = true;

    // Pre-extract forest
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

    // Pre-extract grid query pointers
    std::vector<const double*> Xg_ptrs(G_plus_1);
    for (int g = 0; g < G_plus_1; g++) {
        NumericMatrix Xg = X_grid_list[g];
        Xg_ptrs[g] = REAL(Xg);
    }

    // Accumulators
    std::vector<double> fhat_obs_sum(n, 0.0);
    std::vector<int> fhat_obs_cnt(n, 0);
    std::vector<std::vector<double>> fhat_grid_sum(G_plus_1, std::vector<double>(n, 0.0));
    std::vector<std::vector<int>> fhat_grid_cnt(G_plus_1, std::vector<int>(n, 0));

    // Diagnostic
    int n_split_trees = 0;
    for (int b = 0; b < B; b++) {
        for (int nd = 0; nd < (int)all_sv[b].size(); nd++) {
            if (all_lc[b][nd] != 0 && all_sv[b][nd] == var_col) {
                n_split_trees++;
                break;
            }
        }
    }

    // ========== TREE LOOP ==========
    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();

        // Route honest obs, compute leaf means (ONCE)
        std::unordered_map<int, double> leaf_ysum;
        std::unordered_map<int, int> leaf_ycnt;
        for (int i = 0; i < n; i++) {
            if (!is_honest[i]) continue;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0)
                node = (Xobs_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
            leaf_ysum[node] += yi;
            leaf_ycnt[node]++;
        }
        std::unordered_map<int, double> leaf_mean;
        for (auto& kv : leaf_ycnt)
            if (kv.second > 0)
                leaf_mean[kv.first] = leaf_ysum[kv.first] / kv.second;

        // Route fhat_obs and all grid points
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;

            // fhat_obs
            {
                int node = 0;
                while (lc[node] != 0 || rc[node] != 0)
                    node = (Xobs_ptr[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
                auto it = leaf_mean.find(node);
                if (it != leaf_mean.end()) { fhat_obs_sum[i] += it->second; fhat_obs_cnt[i]++; }
            }

            // Grid points
            for (int g = 0; g < G_plus_1; g++) {
                const double* Xg = Xg_ptrs[g];
                int node = 0;
                while (lc[node] != 0 || rc[node] != 0)
                    node = (Xg[i + n * sv[node]] <= sval[node]) ? lc[node] : rc[node];
                auto it = leaf_mean.find(node);
                if (it != leaf_mean.end()) {
                    fhat_grid_sum[g][i] += it->second;
                    fhat_grid_cnt[g][i]++;
                }
            }
        }
    }
    // ========== END TREE LOOP ==========

    // sigma2_ej: use override if provided, otherwise compute from data
    double sigma2_ej;
    if (sigma2_override > 0.0) {
        sigma2_ej = sigma2_override;
    } else {
        double ss = 0.0;
        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;
            double ej = Xobs_ptr[i + n * var_col] - g_ptr[i];
            ss += ej * ej;
        }
        sigma2_ej = ss / n_hon;
    }

    // Compute AIPW slopes at each interval
    NumericVector slopes(G);
    NumericVector pred_slopes(G);

    for (int g = 0; g < G; g++) {
        double delta_g = grid_points[g + 1] - grid_points[g];
        double phi_sum = 0.0;
        double pred_sum = 0.0;
        int cnt = 0;

        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;

            double fg_hi = (fhat_grid_cnt[g + 1][i] > 0) ?
                fhat_grid_sum[g + 1][i] / fhat_grid_cnt[g + 1][i] : NA_REAL;
            double fg_lo = (fhat_grid_cnt[g][i] > 0) ?
                fhat_grid_sum[g][i] / fhat_grid_cnt[g][i] : NA_REAL;
            double fo = (fhat_obs_cnt[i] > 0) ?
                fhat_obs_sum[i] / fhat_obs_cnt[i] : NA_REAL;

            if (ISNA(fg_hi) || ISNA(fg_lo) || ISNA(fo)) continue;

            double pred_contrast = fg_hi - fg_lo;
            double residual = y_ptr[i] - fo;
            double ej = Xobs_ptr[i + n * var_col] - g_ptr[i];
            double weight = ej / sigma2_ej;
            double phi_g = pred_contrast + weight * residual * delta_g;

            phi_sum += phi_g;
            pred_sum += pred_contrast;
            cnt++;
        }

        slopes[g] = (cnt > 0) ? (phi_sum / cnt) / delta_g : NA_REAL;
        pred_slopes[g] = (cnt > 0) ? (pred_sum / cnt) / delta_g : NA_REAL;
    }

    // Integrate to curve
    NumericVector curve(G_plus_1, 0.0);
    for (int g = 0; g < G; g++) {
        double delta_g = grid_points[g + 1] - grid_points[g];
        curve[g + 1] = curve[g] + (ISNA(slopes[g]) ? 0.0 : slopes[g] * delta_g);
    }

    return List::create(
        Named("slopes") = slopes,
        Named("pred_slopes") = pred_slopes,
        Named("curve") = curve,
        Named("grid") = grid_points,
        Named("sigma2_ej") = sigma2_ej,
        Named("n_hon") = n_hon,
        Named("n_split_trees") = n_split_trees,
        Named("n_trees") = B
    );
}
