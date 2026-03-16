// honest_contrast.cpp — Honest effect estimation for inference forests
//
// Binary predictors: Honest counterfactual routing with conditional leaf means
//   For each obs i in each tree: compute conditional leaf means by X_j group,
//   route factual and counterfactual, extract group-specific means.
//   Augmentation correction applied globally in R (not within leaves).
//
// Continuous predictors: within-leaf binned contrast with augmented numerator
//   Augmentation uses forest-wide predictions precomputed in R.

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
    bool per_leaf_denom = true,
    SEXP cont_fhat_ref = R_NilValue
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

    // Continuous augmentation (precomputed forest-wide predictions)
    bool have_cont_aug = (cont_fhat_ref != R_NilValue);
    NumericMatrix cfr;
    const double* cfr_ptr = nullptr;
    if (have_cont_aug) {
        cfr = as<NumericMatrix>(cont_fhat_ref);
        cfr_ptr = REAL(cfr);
    }

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

    // Accumulators
    std::vector<std::vector<double>> bin_sum(n_bin, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> bin_cnt(n_bin, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> cont_sum(n_cont, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> cont_cnt(n_cont, std::vector<double>(n, 0.0));

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

        // === BINARY: Conditional Leaf Means + Counterfactual Routing ===
        for (int v = 0; v < n_bin; v++) {
            int col = bcols[v];

            // Compute conditional leaf means by X_j group
            struct LeafCond { double sy1=0, sy0=0; int n1=0, n0=0; };
            std::unordered_map<int, LeafCond> leaf_cond;

            for (int i = 0; i < n; i++) {
                if (!is_honest[i]) continue;
                double yi = y_ptr[i];
                if (ISNA(yi)) continue;
                double xi = X_ptr[i + n * col];
                auto& s = leaf_cond[leaf_id[i]];
                if (xi > 0.5) { s.sy1 += yi; s.n1++; }
                else           { s.sy0 += yi; s.n0++; }
            }

            // For each obs: route counterfactual, extract conditional means
            for (int i = 0; i < n; i++) {
                if (!is_honest[i]) continue;
                double yi = y_ptr[i];
                if (ISNA(yi)) continue;
                double xi = X_ptr[i + n * col];
                double xi_flip = (xi > 0.5) ? 0.0 : 1.0;

                // Counterfactual leaf
                int node_cf = 0;
                while (lc[node_cf] != 0 || rc[node_cf] != 0) {
                    double xval = (sv[node_cf] == col) ? xi_flip : X_ptr[i + n * sv[node_cf]];
                    node_cf = (xval <= sval[node_cf]) ? lc[node_cf] : rc[node_cf];
                }

                // Which leaf has x_j=1 conditional mean, which has x_j=0
                int leaf_hi = (xi > 0.5) ? leaf_id[i] : node_cf;
                int leaf_lo = (xi > 0.5) ? node_cf : leaf_id[i];

                auto it_hi = leaf_cond.find(leaf_hi);
                auto it_lo = leaf_cond.find(leaf_lo);
                if (it_hi == leaf_cond.end() || it_lo == leaf_cond.end()) continue;
                if (it_hi->second.n1 == 0 || it_lo->second.n0 == 0) continue;

                double contrast = it_hi->second.sy1 / it_hi->second.n1
                                - it_lo->second.sy0 / it_lo->second.n0;

                bin_sum[v][i] += contrast;
                bin_cnt[v][i] += 1.0;
                bin_global_wsum[v] += contrast;
                bin_global_wcnt[v] += 1.0;
            }
        }

        // === CONTINUOUS: Augmented within-leaf binned contrast ===
        for (int m = 0; m < n_cont; m++) {
            int col = ccols[m];
            double thresh = cthresh[m];

            // DFS: label nodes with first ancestor splitting on col
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

            // Accumulate honest Y, X_k, and augmentation
            struct Sums {
                double sy_hi=0, sy_lo=0, sx_hi=0, sx_lo=0;
                double sf_hi=0, sf_lo=0;
                int nhi=0, nlo=0;
            };
            std::unordered_map<int, Sums> c1_sums, c2_sums;

            for (int i = 0; i < n; i++) {
                if (!is_honest[i]) continue;
                double yi = y_ptr[i];
                if (ISNA(yi)) continue;
                double xi = X_ptr[i + n * col];
                double fi = (have_cont_aug) ? cfr_ptr[i + n * m] : 0.0;
                int anc = xk_anc[leaf_id[i]];
                if (anc >= 0) {
                    auto& s = c1_sums[anc];
                    if (xi >= thresh) { s.sy_hi += yi; s.sx_hi += xi; s.sf_hi += fi; s.nhi++; }
                    else               { s.sy_lo += yi; s.sx_lo += xi; s.sf_lo += fi; s.nlo++; }
                } else {
                    auto& s = c2_sums[leaf_id[i]];
                    if (xi >= thresh) { s.sy_hi += yi; s.sx_hi += xi; s.sf_hi += fi; s.nhi++; }
                    else               { s.sy_lo += yi; s.sx_lo += xi; s.sf_lo += fi; s.nlo++; }
                }
            }

            // Compute augmented contrasts
            auto compute_contrasts = [&](std::unordered_map<int, Sums>& sums_map,
                                         std::unordered_map<int, double>& out_c,
                                         std::unordered_map<int, double>& out_w) {
                for (auto& kv : sums_map) {
                    auto& s = kv.second;
                    if (s.nhi > 0 && s.nlo > 0) {
                        double y_diff = s.sy_hi / s.nhi - s.sy_lo / s.nlo;
                        double f_diff = (have_cont_aug) ? (s.sf_hi / s.nhi - s.sf_lo / s.nlo) : 0.0;
                        double aug_num = y_diff - f_diff;
                        double w = (double)(s.nhi * s.nlo) / (double)(s.nhi + s.nlo);
                        if (per_leaf_denom) {
                            double x_gap = s.sx_hi / s.nhi - s.sx_lo / s.nlo;
                            if (std::abs(x_gap) > 1e-10) {
                                out_c[kv.first] = aug_num / x_gap;
                                out_w[kv.first] = w;
                            }
                        } else {
                            out_c[kv.first] = aug_num;
                            out_w[kv.first] = w;
                        }
                    }
                }
            };

            std::unordered_map<int, double> c1_c, c2_c, c1_w, c2_w;
            compute_contrasts(c1_sums, c1_c, c1_w);
            compute_contrasts(c2_sums, c2_c, c2_w);

            // Accumulate popavg
            for (auto& kv : c1_c) { cont_global_wsum[m] += kv.second * c1_w[kv.first]; cont_global_wcnt[m] += c1_w[kv.first]; }
            for (auto& kv : c2_c) { cont_global_wsum[m] += kv.second * c2_w[kv.first]; cont_global_wcnt[m] += c2_w[kv.first]; }

            // Assign to obs
            for (int i = 0; i < n; i++) {
                int anc = xk_anc[leaf_id[i]];
                double contrast = 0.0, w_r = 0.0;
                bool found = false;
                if (anc >= 0) {
                    auto it = c1_c.find(anc);
                    if (it != c1_c.end()) { contrast = it->second; w_r = c1_w[anc]; found = true; }
                } else {
                    auto it = c2_c.find(leaf_id[i]);
                    if (it != c2_c.end()) { contrast = it->second; w_r = c2_w[leaf_id[i]]; found = true; }
                }
                if (found) { cont_sum[m][i] += contrast * w_r; cont_cnt[m][i] += w_r; }
            }
        }
    }
    // ========== END TREE LOOP ==========

    auto build_out = [&](int nv, std::vector<std::vector<double>>& sums,
                         std::vector<std::vector<double>>& counts,
                         std::vector<double>& gwsum, std::vector<double>& gwcnt) {
        NumericVector pa(nv);
        List om_list(nv);
        for (int v = 0; v < nv; v++) {
            NumericVector om(n);
            for (int i = 0; i < n; i++) {
                om[i] = (counts[v][i] > 0) ? sums[v][i] / counts[v][i] : NA_REAL;
            }
            pa[v] = (gwcnt[v] > 0) ? gwsum[v] / gwcnt[v] : NA_REAL;
            om_list[v] = om;
        }
        return List::create(Named("popavg") = pa, Named("obs_mean") = om_list);
    };

    List bin_out = build_out(n_bin, bin_sum, bin_cnt, bin_global_wsum, bin_global_wcnt);
    List cont_out = build_out(n_cont, cont_sum, cont_cnt, cont_global_wsum, cont_global_wcnt);

    return List::create(Named("binary") = bin_out, Named("continuous") = cont_out);
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
    NumericVector window_hi,
    SEXP fhat_ref_vec = R_NilValue
) {
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

    bool have_aug = (fhat_ref_vec != R_NilValue);
    NumericVector fhat_ref_nv;
    const double* fr_ptr = nullptr;
    if (have_aug) {
        fhat_ref_nv = as<NumericVector>(fhat_ref_vec);
        fr_ptr = REAL(fhat_ref_nv);
    }

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

    std::vector<std::vector<double>> slope_sum(G, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> slope_cnt(G, std::vector<double>(n, 0.0));
    std::vector<double> slope_global_wsum(G, 0.0);
    std::vector<double> slope_global_wcnt(G, 0.0);

    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();
        int n_nodes = (int)all_sv[b].size();

        std::vector<int> leaf_id(n);
        for (int i = 0; i < n; i++) {
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0) {
                double xval = X_ptr[i + n * sv[node]];
                node = (xval <= sval[node]) ? lc[node] : rc[node];
            }
            leaf_id[i] = node;
        }

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

        for (int g = 0; g < G; g++) {
            double mid = mids[g];
            double wl = wlo[g];
            double wh = whi[g];

            struct Sums {
                double sy_hi=0, sy_lo=0, sx_hi=0, sx_lo=0;
                double sf_hi=0, sf_lo=0;
                int nhi=0, nlo=0;
            };
            std::unordered_map<int, Sums> region_sums;

            for (int i = 0; i < n; i++) {
                if (!is_honest[i]) continue;
                double yi = y_ptr[i];
                if (ISNA(yi)) continue;
                double xi = X_ptr[i + n * col];
                if (xi < wl || xi > wh) continue;

                double fi = (have_aug) ? fr_ptr[i] : 0.0;
                int anc = xj_anc[leaf_id[i]];
                int region_key = (anc >= 0) ? anc : -(leaf_id[i] + 1);

                auto& s = region_sums[region_key];
                if (xi >= mid) { s.sy_hi += yi; s.sx_hi += xi; s.sf_hi += fi; s.nhi++; }
                else            { s.sy_lo += yi; s.sx_lo += xi; s.sf_lo += fi; s.nlo++; }
            }

            std::unordered_map<int, double> region_slopes;
            std::unordered_map<int, double> region_weights;
            for (auto& kv : region_sums) {
                auto& s = kv.second;
                if (s.nhi > 0 && s.nlo > 0) {
                    double x_gap = s.sx_hi / s.nhi - s.sx_lo / s.nlo;
                    if (std::abs(x_gap) > 1e-10) {
                        double y_diff = s.sy_hi / s.nhi - s.sy_lo / s.nlo;
                        double f_diff = (have_aug) ? (s.sf_hi / s.nhi - s.sf_lo / s.nlo) : 0.0;
                        region_slopes[kv.first] = (y_diff - f_diff) / x_gap;
                        region_weights[kv.first] = (double)(s.nhi * s.nlo) / (double)(s.nhi + s.nlo);
                    }
                }
            }

            for (auto& kv : region_slopes) {
                double w = region_weights[kv.first];
                slope_global_wsum[g] += kv.second * w;
                slope_global_wcnt[g] += w;
            }

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

    NumericVector pa(G);
    List om_list(G);
    for (int g = 0; g < G; g++) {
        NumericVector om(n);
        for (int i = 0; i < n; i++)
            om[i] = (slope_cnt[g][i] > 0) ? slope_sum[g][i] / slope_cnt[g][i] : NA_REAL;
        pa[g] = (slope_global_wcnt[g] > 0) ? slope_global_wsum[g] / slope_global_wcnt[g] : NA_REAL;
        om_list[g] = om;
    }

    return List::create(Named("popavg") = pa, Named("obs_mean") = om_list);
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
    double int_global_wsum = 0.0;
    double int_global_wcnt = 0.0;

    for (int b = 0; b < B; b++) {
        const int* sv = all_sv[b].data();
        const double* sval = all_sval[b].data();
        const int* lc = all_lc[b].data();
        const int* rc = all_rc[b].data();

        std::vector<int> leaf_id(n);
        for (int i = 0; i < n; i++) {
            int node = 0;
            while (lc[node] != 0 || rc[node] != 0) {
                double xval = X_ptr[i + n * sv[node]];
                node = (xval <= sval[node]) ? lc[node] : rc[node];
            }
            leaf_id[i] = node;
        }

        struct Cell {
            double sy11=0, sy10=0, sy01=0, sy00=0;
            double sx11=0, sx10=0, sx01=0, sx00=0;
            int n11=0, n10=0, n01=0, n00=0;
        };
        std::unordered_map<int, Cell> leaf_cells;

        for (int i = 0; i < n; i++) {
            if (!is_honest[i]) continue;
            double yi = y_ptr[i];
            if (ISNA(yi)) continue;
            double xb = X_ptr[i + n * bin_col];
            double xc = X_ptr[i + n * cont_col];
            auto& c = leaf_cells[leaf_id[i]];
            if (xb > 0.5) {
                if (xc >= cont_thresh) { c.sy11 += yi; c.sx11 += xc; c.n11++; }
                else                    { c.sy10 += yi; c.sx10 += xc; c.n10++; }
            } else {
                if (xc >= cont_thresh) { c.sy01 += yi; c.sx01 += xc; c.n01++; }
                else                    { c.sy00 += yi; c.sx00 += xc; c.n00++; }
            }
        }

        std::unordered_map<int, double> leaf_int;
        std::unordered_map<int, double> leaf_int_w;
        for (auto& kv : leaf_cells) {
            auto& c = kv.second;
            if (c.n11 > 0 && c.n10 > 0 && c.n01 > 0 && c.n00 > 0) {
                double my11 = c.sy11/c.n11, my10 = c.sy10/c.n10;
                double my01 = c.sy01/c.n01, my00 = c.sy00/c.n00;
                double raw_int = (my11 - my10) - (my01 - my00);
                double sx_hi = c.sx11 + c.sx01;
                int n_hi = c.n11 + c.n01;
                double sx_lo = c.sx10 + c.sx00;
                int n_lo = c.n10 + c.n00;
                double x_gap = sx_hi / n_hi - sx_lo / n_lo;
                if (std::abs(x_gap) > 1e-10) {
                    leaf_int[kv.first] = raw_int / x_gap;
                    double inv_w = 1.0/c.n11 + 1.0/c.n10 + 1.0/c.n01 + 1.0/c.n00;
                    leaf_int_w[kv.first] = 1.0 / inv_w;
                }
            }
        }

        for (auto& kv : leaf_int) {
            double w = leaf_int_w[kv.first];
            int_global_wsum += kv.second * w;
            int_global_wcnt += w;
        }

        for (int i = 0; i < n; i++) {
            auto it = leaf_int.find(leaf_id[i]);
            if (it != leaf_int.end()) {
                double w = leaf_int_w[leaf_id[i]];
                obs_sum[i] += it->second * w;
                obs_cnt[i] += w;
            }
        }
    }

    double popavg = (int_global_wcnt > 0) ? int_global_wsum / int_global_wcnt : NA_REAL;

    NumericVector om(n);
    for (int i = 0; i < n; i++)
        om[i] = (obs_cnt[i] > 0) ? obs_sum[i] / obs_cnt[i] : NA_REAL;

    return List::create(Named("popavg") = popavg, Named("obs_mean") = om);
}
