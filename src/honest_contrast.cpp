// honest_contrast.cpp — Honest effect estimation for random forests
//
// All estimators use within-leaf honest contrasts. No counterfactual routing.
//
// Binary predictors: within-leaf group mean difference (X_k=1 vs X_k=0)
//   Case 1: tree split on X_k → pooled-downstream contrast at split node
//   Case 2: tree didn't split → within-leaf group means
//
// Continuous predictors: within-leaf binned contrast
//   Bin honest obs into X_k >= threshold vs X_k < threshold
//   Contrast = mean(Y_honest | X_k >= thresh) - mean(Y_honest | X_k < thresh)
//   Uses all trees, all leaves. Same logic as binary but with binned groups.
//
// honest_all: batch all binary + continuous contrasts in one pass per tree.

#include <Rcpp.h>
#include <vector>
#include <unordered_map>
using namespace Rcpp;

// [[Rcpp::export]]
List honest_all(
    List forest,
    NumericMatrix X_num,
    NumericVector y_honest,
    IntegerVector honest_idx,
    IntegerVector bin_cols,
    IntegerVector cont_cols,
    NumericVector cont_thresh,
    bool per_leaf_denom = true
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_num.nrow();
    int n_bin = bin_cols.size();
    int n_cont = cont_cols.size();

    const double* X_ptr = REAL(X_num);
    const double* y_ptr = REAL(y_honest);

    std::vector<bool> is_honest(n, false);
    for (int j = 0; j < honest_idx.size(); j++)
        is_honest[honest_idx[j] - 1] = true;

    // Pre-extract forest structure
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

    std::vector<int> bcols(bin_cols.begin(), bin_cols.end());
    std::vector<int> ccols(cont_cols.begin(), cont_cols.end());
    std::vector<double> cthresh(cont_thresh.begin(), cont_thresh.end());

    // Accumulators — per-observation (for obs_mean output)
    std::vector<std::vector<double>> bin_sum(n_bin, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> bin_cnt(n_bin, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> cont_sum(n_cont, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> cont_cnt(n_cont, std::vector<double>(n, 0.0));

    // Forest-wide weighted sums for popavg — each unique contrast contributes once per tree
    std::vector<double> bin_global_wsum(n_bin, 0.0);
    std::vector<double> bin_global_wcnt(n_bin, 0.0);
    std::vector<double> cont_global_wsum(n_cont, 0.0);
    std::vector<double> cont_global_wcnt(n_cont, 0.0);

    // ========== TREE LOOP ==========
    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();
        int n_nodes = (int)all_sv[b].size();

        // --- Walk all obs to leaves (once per tree) ---
        std::vector<int> leaf_id(n);
        for (int i = 0; i < n; i++) {
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0) {
                double xval = X_ptr[i + n * sv[node]];
                node = (xval <= sval[node]) ? lc[node] : rc[node];
            }
            leaf_id[i] = node;
        }

        // === BINARY CONTRASTS ===
        for (int v = 0; v < n_bin; v++) {
            int col = bcols[v];

            // DFS to label every node with first ancestor that splits on this col
            std::vector<int> x6_anc(n_nodes, -1);
            {
                struct DE { int node; int anc; };
                std::vector<DE> stk;
                stk.push_back({0, -1});
                while (!stk.empty()) {
                    auto e = stk.back(); stk.pop_back();
                    x6_anc[e.node] = e.anc;
                    if (lc[e.node] != 0 || rc[e.node] != 0) {
                        if (sv[e.node] == col && e.anc < 0) {
                            stk.push_back({lc[e.node], e.node});
                            stk.push_back({rc[e.node], e.node});
                        } else {
                            stk.push_back({lc[e.node], e.anc});
                            stk.push_back({rc[e.node], e.anc});
                        }
                    }
                }
            }

            // Accumulate honest Y by region and binary group
            struct Sums { double s1=0, s0=0; int n1=0, n0=0; };
            std::unordered_map<int, Sums> c1_sums, c2_sums;

            for (int i = 0; i < n; i++) {
                if (!is_honest[i]) continue;
                double yi = y_ptr[i];
                if (ISNA(yi)) continue;
                double xi = X_ptr[i + n * col];
                int anc = x6_anc[leaf_id[i]];
                if (anc >= 0) {
                    auto& s = c1_sums[anc];
                    if (xi > 0.5) { s.s1 += yi; s.n1++; }
                    else           { s.s0 += yi; s.n0++; }
                } else {
                    auto& s = c2_sums[leaf_id[i]];
                    if (xi > 0.5) { s.s1 += yi; s.n1++; }
                    else           { s.s0 += yi; s.n0++; }
                }
            }

            // Compute contrasts and inverse-variance weights
            std::unordered_map<int, double> c1_c, c2_c;
            std::unordered_map<int, double> c1_w, c2_w;  // harmonic weights
            for (auto& kv : c1_sums) {
                auto& s = kv.second;
                if (s.n1 > 0 && s.n0 > 0) {
                    c1_c[kv.first] = s.s1 / s.n1 - s.s0 / s.n0;
                    c1_w[kv.first] = (double)(s.n1 * s.n0) / (double)(s.n1 + s.n0);
                }
            }
            for (auto& kv : c2_sums) {
                auto& s = kv.second;
                if (s.n1 > 0 && s.n0 > 0) {
                    c2_c[kv.first] = s.s1 / s.n1 - s.s0 / s.n0;
                    c2_w[kv.first] = (double)(s.n1 * s.n0) / (double)(s.n1 + s.n0);
                }
            }

            // Accumulate into forest-wide popavg — each contrast once
            for (auto& kv : c1_c) {
                double w = c1_w[kv.first];
                bin_global_wsum[v] += kv.second * w;
                bin_global_wcnt[v] += w;
            }
            for (auto& kv : c2_c) {
                double w = c2_w[kv.first];
                bin_global_wsum[v] += kv.second * w;
                bin_global_wcnt[v] += w;
            }

            // Assign to obs — for per-observation output
            for (int i = 0; i < n; i++) {
                int anc = x6_anc[leaf_id[i]];
                bool found = false;
                double contrast = 0.0;
                double w_region = 0.0;
                if (anc >= 0) {
                    auto it = c1_c.find(anc);
                    if (it != c1_c.end()) { contrast = it->second; w_region = c1_w[anc]; found = true; }
                } else {
                    auto it = c2_c.find(leaf_id[i]);
                    if (it != c2_c.end()) { contrast = it->second; w_region = c2_w[leaf_id[i]]; found = true; }
                }
                if (found) {
                    bin_sum[v][i] += contrast * w_region;
                    bin_cnt[v][i] += w_region;
                }
            }
        }

        // === CONTINUOUS CONTRASTS (within-leaf binning) ===
        // Same logic as binary: within each leaf (or pooled-downstream at a split node),
        // partition honest obs into X_k >= threshold vs X_k < threshold.
        // Per-leaf slope = (mean_Y_hi - mean_Y_lo) / (mean_Xk_hi - mean_Xk_lo)
        // Denominator from honest X values only (no Y dependence).
        // Already on per-unit scale — no external span division needed.
        for (int m = 0; m < n_cont; m++) {
            int col = ccols[m];
            double thresh = cthresh[m];

            // DFS to label nodes with first ancestor that splits on this col
            std::vector<int> xk_anc(n_nodes, -1);
            {
                struct DE { int node; int anc; };
                std::vector<DE> stk;
                stk.push_back({0, -1});
                while (!stk.empty()) {
                    auto e = stk.back(); stk.pop_back();
                    xk_anc[e.node] = e.anc;
                    if (lc[e.node] != 0 || rc[e.node] != 0) {
                        if (sv[e.node] == col && e.anc < 0) {
                            stk.push_back({lc[e.node], e.node});
                            stk.push_back({rc[e.node], e.node});
                        } else {
                            stk.push_back({lc[e.node], e.anc});
                            stk.push_back({rc[e.node], e.anc});
                        }
                    }
                }
            }

            // Accumulate honest Y AND X_k by region and binned group
            struct Sums {
                double sy_hi=0, sy_lo=0, sx_hi=0, sx_lo=0;
                int nhi=0, nlo=0;
            };
            std::unordered_map<int, Sums> c1_sums, c2_sums;

            for (int i = 0; i < n; i++) {
                if (!is_honest[i]) continue;
                double yi = y_ptr[i];
                if (ISNA(yi)) continue;
                double xi = X_ptr[i + n * col];
                int anc = xk_anc[leaf_id[i]];
                if (anc >= 0) {
                    auto& s = c1_sums[anc];
                    if (xi >= thresh) { s.sy_hi += yi; s.sx_hi += xi; s.nhi++; }
                    else               { s.sy_lo += yi; s.sx_lo += xi; s.nlo++; }
                } else {
                    auto& s = c2_sums[leaf_id[i]];
                    if (xi >= thresh) { s.sy_hi += yi; s.sx_hi += xi; s.nhi++; }
                    else               { s.sy_lo += yi; s.sx_lo += xi; s.nlo++; }
                }
            }

            // Compute contrasts and region sizes: per-leaf slope if per_leaf_denom, raw Y diff otherwise
            std::unordered_map<int, double> c1_c, c2_c;
            std::unordered_map<int, double> c1_w, c2_w;  // harmonic weights
            for (auto& kv : c1_sums) {
                auto& s = kv.second;
                if (s.nhi > 0 && s.nlo > 0) {
                    double y_diff = s.sy_hi / s.nhi - s.sy_lo / s.nlo;
                    double w = (double)(s.nhi * s.nlo) / (double)(s.nhi + s.nlo);
                    if (per_leaf_denom) {
                        double x_gap = s.sx_hi / s.nhi - s.sx_lo / s.nlo;
                        if (std::abs(x_gap) > 1e-10) {
                            c1_c[kv.first] = y_diff / x_gap;
                            c1_w[kv.first] = w;
                        }
                    } else {
                        c1_c[kv.first] = y_diff;
                        c1_w[kv.first] = w;
                    }
                }
            }
            for (auto& kv : c2_sums) {
                auto& s = kv.second;
                if (s.nhi > 0 && s.nlo > 0) {
                    double y_diff = s.sy_hi / s.nhi - s.sy_lo / s.nlo;
                    double w = (double)(s.nhi * s.nlo) / (double)(s.nhi + s.nlo);
                    if (per_leaf_denom) {
                        double x_gap = s.sx_hi / s.nhi - s.sx_lo / s.nlo;
                        if (std::abs(x_gap) > 1e-10) {
                            c2_c[kv.first] = y_diff / x_gap;
                            c2_w[kv.first] = w;
                        }
                    } else {
                        c2_c[kv.first] = y_diff;
                        c2_w[kv.first] = w;
                    }
                }
            }

            // Accumulate into forest-wide popavg — each contrast once
            for (auto& kv : c1_c) {
                double w = c1_w[kv.first];
                cont_global_wsum[m] += kv.second * w;
                cont_global_wcnt[m] += w;
            }
            for (auto& kv : c2_c) {
                double w = c2_w[kv.first];
                cont_global_wsum[m] += kv.second * w;
                cont_global_wcnt[m] += w;
            }

            // Assign to obs — for per-observation output
            for (int i = 0; i < n; i++) {
                int anc = xk_anc[leaf_id[i]];
                bool found = false;
                double contrast = 0.0;
                double w_region = 0.0;
                if (anc >= 0) {
                    auto it = c1_c.find(anc);
                    if (it != c1_c.end()) { contrast = it->second; w_region = c1_w[anc]; found = true; }
                } else {
                    auto it = c2_c.find(leaf_id[i]);
                    if (it != c2_c.end()) { contrast = it->second; w_region = c2_w[leaf_id[i]]; found = true; }
                }
                if (found) {
                    cont_sum[m][i] += contrast * w_region;
                    cont_cnt[m][i] += w_region;
                }
            }
        }
    }
    // ========== END TREE LOOP ==========

    // Build output
    auto build_out = [&](int nv, std::vector<std::vector<double>>& sums,
                         std::vector<std::vector<double>>& counts,
                         std::vector<double>& gwsum, std::vector<double>& gwcnt) {
        NumericVector pa(nv);
        List om_list(nv);
        for (int v = 0; v < nv; v++) {
            NumericVector om(n);
            for (int i = 0; i < n; i++) {
                if (counts[v][i] > 0) {
                    om[i] = sums[v][i] / counts[v][i];
                } else {
                    om[i] = NA_REAL;
                }
            }
            pa[v] = (gwcnt[v] > 0) ? gwsum[v] / gwcnt[v] : NA_REAL;
            om_list[v] = om;
        }
        return List::create(Named("popavg") = pa, Named("obs_mean") = om_list);
    };

    List bin_out = build_out(n_bin, bin_sum, bin_cnt, bin_global_wsum, bin_global_wcnt);
    List cont_out = build_out(n_cont, cont_sum, cont_cnt, cont_global_wsum, cont_global_wcnt);

    return List::create(
        Named("binary") = bin_out,
        Named("continuous") = cont_out
    );
}


// [[Rcpp::export]]
List honest_curve(
    List forest,
    NumericMatrix X_num,
    NumericVector y_honest,
    IntegerVector honest_idx,
    int col,
    NumericVector midpoints,
    NumericVector window_lo,
    NumericVector window_hi
) {
    // Curve-based estimation for one continuous predictor.
    // For each grid interval g with midpoint midpoints[g]:
    //   - Find estimation region (Case 1: pooled-downstream at first X_j split,
    //     Case 2: leaf if no split)
    //   - Restrict honest obs in region to window [window_lo[g], window_hi[g]]
    //   - Split at midpoints[g]: upper = X_j >= midpoint, lower = X_j < midpoint
    //   - Compute slope = (Ybar_upper - Ybar_lower) / (Xbar_j_upper - Xbar_j_lower)
    //   - Skip if either group empty
    //
    // Output: per-obs slope at each grid interval, popavg slopes.

    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_num.nrow();
    int G = midpoints.size();

    const double* X_ptr = REAL(X_num);
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

    std::vector<double> mids(midpoints.begin(), midpoints.end());
    std::vector<double> wlo(window_lo.begin(), window_lo.end());
    std::vector<double> whi(window_hi.begin(), window_hi.end());

    // Accumulators: per-obs, per-grid-interval
    std::vector<std::vector<double>> slope_sum(G, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> slope_cnt(G, std::vector<double>(n, 0.0));

    // Forest-wide weighted sums for popavg slopes
    std::vector<double> slope_global_wsum(G, 0.0);
    std::vector<double> slope_global_wcnt(G, 0.0);

    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();
        int n_nodes = (int)all_sv[b].size();

        // Route all obs to leaves
        std::vector<int> leaf_id(n);
        for (int i = 0; i < n; i++) {
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0) {
                double xval = X_ptr[i + n * sv[node]];
                node = (xval <= sval[node]) ? lc[node] : rc[node];
            }
            leaf_id[i] = node;
        }

        // DFS: label nodes with first ancestor that splits on col
        std::vector<int> xj_anc(n_nodes, -1);
        {
            struct DE { int node; int anc; };
            std::vector<DE> stk;
            stk.push_back({0, -1});
            while (!stk.empty()) {
                auto e = stk.back(); stk.pop_back();
                xj_anc[e.node] = e.anc;
                if (lc[e.node] != 0 || rc[e.node] != 0) {
                    if (sv[e.node] == col && e.anc < 0) {
                        stk.push_back({lc[e.node], e.node});
                        stk.push_back({rc[e.node], e.node});
                    } else {
                        stk.push_back({lc[e.node], e.anc});
                        stk.push_back({rc[e.node], e.anc});
                    }
                }
            }
        }

        // For each grid interval, accumulate windowed honest sums by region
        for (int g = 0; g < G; g++) {
            double mid = mids[g];
            double wl = wlo[g];
            double wh = whi[g];

            // Accumulate by region key: positive = Case 1 ancestor node, negative = -(leaf+1) for Case 2
            struct Sums {
                double sy_hi=0, sy_lo=0, sx_hi=0, sx_lo=0;
                int nhi=0, nlo=0;
            };
            std::unordered_map<int, Sums> region_sums;

            for (int i = 0; i < n; i++) {
                if (!is_honest[i]) continue;
                double yi = y_ptr[i];
                if (ISNA(yi)) continue;
                double xi = X_ptr[i + n * col];

                // Window filter
                if (xi < wl || xi > wh) continue;

                int anc = xj_anc[leaf_id[i]];
                int region_key = (anc >= 0) ? anc : -(leaf_id[i] + 1);

                auto& s = region_sums[region_key];
                if (xi >= mid) { s.sy_hi += yi; s.sx_hi += xi; s.nhi++; }
                else            { s.sy_lo += yi; s.sx_lo += xi; s.nlo++; }
            }

            // Compute per-region slopes and harmonic weights
            std::unordered_map<int, double> region_slopes;
            std::unordered_map<int, double> region_weights;
            for (auto& kv : region_sums) {
                auto& s = kv.second;
                if (s.nhi > 0 && s.nlo > 0) {
                    double x_gap = s.sx_hi / s.nhi - s.sx_lo / s.nlo;
                    if (std::abs(x_gap) > 1e-10) {
                        region_slopes[kv.first] = (s.sy_hi / s.nhi - s.sy_lo / s.nlo) / x_gap;
                        region_weights[kv.first] = (double)(s.nhi * s.nlo) / (double)(s.nhi + s.nlo);
                    }
                }
            }

            // Accumulate into forest-wide popavg — each contrast once
            for (auto& kv : region_slopes) {
                double w = region_weights[kv.first];
                slope_global_wsum[g] += kv.second * w;
                slope_global_wcnt[g] += w;
            }

            // Assign to obs — for per-observation output
            for (int i = 0; i < n; i++) {
                int anc = xj_anc[leaf_id[i]];
                int region_key = (anc >= 0) ? anc : -(leaf_id[i] + 1);
                auto it = region_slopes.find(region_key);
                if (it != region_slopes.end()) {
                    double w = region_weights[region_key];
                    slope_sum[g][i] += it->second * w;
                    slope_cnt[g][i] += w;
                }
            }
        }
    }

    // Build output
    NumericVector pa(G);
    List om_list(G);
    for (int g = 0; g < G; g++) {
        NumericVector om(n);
        for (int i = 0; i < n; i++) {
            if (slope_cnt[g][i] > 0) {
                om[i] = slope_sum[g][i] / slope_cnt[g][i];
            } else {
                om[i] = NA_REAL;
            }
        }
        pa[g] = (slope_global_wcnt[g] > 0) ? slope_global_wsum[g] / slope_global_wcnt[g] : NA_REAL;
        om_list[g] = om;
    }

    return List::create(
        Named("popavg_slopes") = pa,
        Named("obs_slopes") = om_list
    );
}


// [[Rcpp::export]]
List honest_interaction_2x2(
    List forest,
    NumericMatrix X_num,
    NumericVector y_honest,
    IntegerVector honest_idx,
    int bin_col,
    int cont_col,
    double cont_thresh
) {
    // Within-leaf 2x2 interaction: bin_col (binary 0/1) x cont_col (>= thresh vs < thresh)
    //
    // For each tree, for each leaf (or pooled-downstream region):
    //   Compute 4 cell means from honest obs:
    //     m_11 = mean(Y | bin=1, cont >= thresh)
    //     m_10 = mean(Y | bin=1, cont < thresh)
    //     m_01 = mean(Y | bin=0, cont >= thresh)
    //     m_00 = mean(Y | bin=0, cont < thresh)
    //   Interaction = (m_11 - m_10) - (m_01 - m_00)
    //   i.e., the cont effect among bin=1 minus the cont effect among bin=0
    //
    // Every leaf contributes if all 4 cells have honest obs.
    // Skip if any cell is empty.

    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_num.nrow();

    const double* X_ptr = REAL(X_num);
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

    std::vector<double> obs_sum(n, 0.0);
    std::vector<double> obs_cnt(n, 0.0);

    // Forest-wide weighted sum for popavg
    double int_global_wsum = 0.0;
    double int_global_wcnt = 0.0;

    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();

        // Walk all obs to leaves
        std::vector<int> leaf_id(n);
        for (int i = 0; i < n; i++) {
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0) {
                double xval = X_ptr[i + n * sv[node]];
                node = (xval <= sval[node]) ? lc[node] : rc[node];
            }
            leaf_id[i] = node;
        }

        // DFS: label nodes with first ancestor that splits on EITHER bin_col or cont_col
        // We need the coarsest region where both variables can be contrasted.
        // Use the leaf directly — every leaf gets a 2x2 contrast from honest obs.
        // No Case 1 / Case 2 distinction needed for the interaction itself —
        // we just need the 4 cell means within whatever region the obs lands in.
        //
        // Simplest correct approach: use the leaf. Partition honest obs in each
        // leaf into the 4 cells. Compute difference-in-differences.

        struct Cell4 {
            double sy11=0,sy10=0,sy01=0,sy00=0;
            double sx11=0,sx10=0,sx01=0,sx00=0;
            int n11=0,n10=0,n01=0,n00=0;
        };
        std::unordered_map<int, Cell4> leaf_cells;

        for (int i = 0; i < n; i++) {
            if (!is_honest[i]) continue;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            double bi = X_ptr[i + n * bin_col];
            double ci = X_ptr[i + n * cont_col];
            int b_group = (bi > 0.5) ? 1 : 0;
            int c_group = (ci >= cont_thresh) ? 1 : 0;

            auto& cell = leaf_cells[leaf_id[i]];
            if (b_group == 1 && c_group == 1)      { cell.sy11 += yi; cell.sx11 += ci; cell.n11++; }
            else if (b_group == 1 && c_group == 0)  { cell.sy10 += yi; cell.sx10 += ci; cell.n10++; }
            else if (b_group == 0 && c_group == 1)  { cell.sy01 += yi; cell.sx01 += ci; cell.n01++; }
            else                                     { cell.sy00 += yi; cell.sx00 += ci; cell.n00++; }
        }

        // Compute per-unit interaction per leaf
        // Numerator: 4-cell difference-in-differences on Y
        // Denominator: 2-group x2 gap (same as main effect denominator)
        //   = mean(x2 | x2 >= thresh, in leaf) - mean(x2 | x2 < thresh, in leaf)
        //   pooled across both x6 groups
        std::unordered_map<int, double> leaf_int;
        std::unordered_map<int, double> leaf_int_w;
        for (auto& kv : leaf_cells) {
            auto& c = kv.second;
            if (c.n11 > 0 && c.n10 > 0 && c.n01 > 0 && c.n00 > 0) {
                double my11 = c.sy11/c.n11, my10 = c.sy10/c.n10;
                double my01 = c.sy01/c.n01, my00 = c.sy00/c.n00;
                double raw_int = (my11 - my10) - (my01 - my00);
                // x2 gap pooled across x6 groups (same as main effect denominator)
                double sx_hi_total = c.sx11 + c.sx01;
                int n_hi_total = c.n11 + c.n01;
                double sx_lo_total = c.sx10 + c.sx00;
                int n_lo_total = c.n10 + c.n00;
                double x_gap = sx_hi_total / n_hi_total - sx_lo_total / n_lo_total;
                if (std::abs(x_gap) > 1e-10) {
                    leaf_int[kv.first] = raw_int / x_gap;
                    // Inverse-variance weight: 1/(1/n11 + 1/n10 + 1/n01 + 1/n00)
                    double inv_w = 1.0/c.n11 + 1.0/c.n10 + 1.0/c.n01 + 1.0/c.n00;
                    leaf_int_w[kv.first] = 1.0 / inv_w;
                }
            }
        }

        // Accumulate into forest-wide popavg — each contrast once
        for (auto& kv : leaf_int) {
            double w = leaf_int_w[kv.first];
            int_global_wsum += kv.second * w;
            int_global_wcnt += w;
        }

        // Assign to obs — for per-observation output
        for (int i = 0; i < n; i++) {
            auto it = leaf_int.find(leaf_id[i]);
            if (it != leaf_int.end()) {
                double w = leaf_int_w[leaf_id[i]];
                obs_sum[i] += it->second * w;
                obs_cnt[i] += w;
            }
        }
    }

    // Output
    NumericVector obs_mean(n);
    for (int i = 0; i < n; i++) {
        if (obs_cnt[i] > 0) {
            obs_mean[i] = obs_sum[i] / obs_cnt[i];
        } else {
            obs_mean[i] = NA_REAL;
        }
    }
    double popavg = (int_global_wcnt > 0) ? int_global_wsum / int_global_wcnt : NA_REAL;

    return List::create(
        Named("popavg") = popavg,
        Named("obs_mean") = obs_mean
    );
}
