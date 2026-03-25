// pasr_extract.cpp — PASR batch extraction and marginal prediction
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
NumericMatrix pasr_extract_all_marginal_cpp(
    List cache_list,           // R-length list of caches
    List Y_syn_list,           // R-length list of Y_syn vectors
    List X_cf_list,            // n_queries-length list of X_cf matrices (n x p, col-major)
    NumericMatrix omega_mat,   // n x n_queries matrix of omega weights
    int n_threads = 1
) {
    int R = cache_list.size();
    int n_queries = X_cf_list.size();

    // Pre-extract X_cf pointers
    std::vector<const double*> xcf_ptrs(n_queries);
    std::vector<int> xcf_nrow(n_queries);
    for (int q = 0; q < n_queries; q++) {
        NumericMatrix Xm = as<NumericMatrix>(X_cf_list[q]);
        xcf_ptrs[q] = REAL(Xm);
        xcf_nrow[q] = Xm.nrow();
    }
    const double* omega_ptr = REAL(omega_mat);
    int n_omega = omega_mat.nrow();

    // Pre-extract cache data
    struct CDat {
        int B, n, p, n_hon;
        const int* honest_idx;
        const int* tree_offsets;
        const int* flat_sv;
        const double* flat_sval;
        const int* flat_lc;
        const int* flat_rc;
        const double* leaf_mean_flat;
        const double* fhat_obs;
    };

    // 2 directions per replicate: A_AB, A_BA
    std::vector<std::vector<CDat>> all_cd(R, std::vector<CDat>(2));
    std::vector<std::vector<List>> holders(R, std::vector<List>(2));
    const char* dir_names[2] = {"A_AB", "A_BA"};

    // Pre-extract Y_syn pointers
    std::vector<const double*> ysyn_ptrs(R);
    std::vector<NumericVector> ysyn_holders(R);
    for (int r = 0; r < R; r++) {
        ysyn_holders[r] = as<NumericVector>(Y_syn_list[r]);
        ysyn_ptrs[r] = REAL(ysyn_holders[r]);
    }

    for (int r = 0; r < R; r++) {
        List rep = cache_list[r];
        for (int d = 0; d < 2; d++) {
            List c = rep[dir_names[d]];
            holders[r][d] = c;
            CDat& cd = all_cd[r][d];
            cd.B = as<int>(c["B"]);
            cd.n = as<int>(c["n"]);
            cd.p = as<int>(c["p"]);
            cd.n_hon = as<int>(c["n_hon"]);
            cd.honest_idx = INTEGER(as<IntegerVector>(c["honest_idx"]));
            cd.tree_offsets = INTEGER(as<IntegerVector>(c["tree_offsets"]));
            cd.flat_sv = INTEGER(as<IntegerVector>(c["flat_sv"]));
            cd.flat_sval = REAL(as<NumericVector>(c["flat_sval"]));
            cd.flat_lc = INTEGER(as<IntegerVector>(c["flat_lc"]));
            cd.flat_rc = INTEGER(as<IntegerVector>(c["flat_rc"]));
            cd.leaf_mean_flat = REAL(as<NumericVector>(c["leaf_mean_flat"]));
            cd.fhat_obs = REAL(as<NumericVector>(c["fhat_obs"]));
        }
    }

    // Output: R x n_queries
    std::vector<double> result_flat(R * n_queries, NA_REAL);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int r = 0; r < R; r++) {
        const double* Y_syn = ysyn_ptrs[r];

        for (int q = 0; q < n_queries; q++) {
            const double* Xcf = xcf_ptrs[q];
            int n_cf = xcf_nrow[q];
            double phi_sum = 0.0;
            int phi_cnt = 0;

            for (int d = 0; d < 2; d++) {
                const CDat& cd = all_cd[r][d];
                int B = cd.B, p = cd.p, n_hon = cd.n_hon; (void)p;
                int n_full = cd.n; (void)n_full;

                // Route all observations through trees with X_cf
                // Then compute AIPW for honest obs only
                for (int j = 0; j < n_hon; j++) {
                    int k = cd.honest_idx[j] - 1;  // 0-based obs index

                    // fhat_cf: route obs k through all trees with X_cf
                    double cf_sum = 0.0; int cf_cnt = 0;
                    for (int bb = 0; bb < B; bb++) {
                        int off = cd.tree_offsets[bb];
                        int node = 0;
                        while (cd.flat_lc[off + node] != 0 || cd.flat_rc[off + node] != 0) {
                            int var = cd.flat_sv[off + node];
                            double val = Xcf[k + n_cf * var];
                            node = (val <= cd.flat_sval[off + node]) ?
                                   cd.flat_lc[off + node] : cd.flat_rc[off + node];
                        }
                        double lm = cd.leaf_mean_flat[off + node];
                        if (lm != -1e308) { cf_sum += lm; cf_cnt++; }
                    }

                    double fhat_cf = (cf_cnt > 0) ? cf_sum / cf_cnt : NA_REAL;
                    double fhat_obs_j = cd.fhat_obs[j];
                    if (ISNA(fhat_cf) || ISNA(fhat_obs_j)) continue;

                    double R_k = Y_syn[k] - fhat_obs_j;
                    double omega_k = omega_ptr[k + n_omega * q];

                    phi_sum += fhat_cf + omega_k * R_k;
                    phi_cnt++;
                }
            }

            result_flat[r * n_queries + q] = (phi_cnt > 0) ? phi_sum / phi_cnt : NA_REAL;
        }
    }

    // Convert to R matrix
    NumericMatrix result(R, n_queries);
    for (int r = 0; r < R; r++)
        for (int q = 0; q < n_queries; q++)
            result(r, q) = result_flat[r * n_queries + q];
    return result;
}


// ============================================================
// pasr_extract_all_level_curve_cpp
//
// Batch level curve: given R replicates, returns R x (G+1) matrix.
// Each entry is the AIPW-adjusted level at that grid point.
// ============================================================
// [[Rcpp::export]]
NumericMatrix pasr_extract_all_level_curve_cpp(
    List cache_list, List Y_syn_list,
    NumericVector ghat, NumericVector x_var,
    int var_col, NumericVector grid_points,
    int n_threads = 1
) {
    int R = cache_list.size();
    int G_plus_1 = grid_points.size();
    const double* grid_ptr = REAL(grid_points);
    const double* g_ptr = REAL(ghat);
    const double* xv_ptr = REAL(x_var);

    struct CDatL {
        int B, n, p, n_hon;
        const int* honest_idx;
        const int* tree_offsets;
        const int* flat_sv;
        const double* flat_sval;
        const int* flat_lc;
        const int* flat_rc;
        const double* X_row_flat;
        const int* obs_leaf_flat;
        const double* leaf_mean_flat;
        const int* splits_on;
        const double* fhat_obs;
    };

    // Pre-extract: 2 dirs per replicate (A_AB, A_BA)
    std::vector<std::vector<CDatL>> all_cd(R, std::vector<CDatL>(2));
    std::vector<std::vector<List>> holders(R, std::vector<List>(2));
    const char* dir_names[2] = {"A_AB", "A_BA"};

    std::vector<const double*> ysyn_ptrs(R);
    std::vector<NumericVector> ysyn_holders(R);
    for (int r = 0; r < R; r++) {
        ysyn_holders[r] = as<NumericVector>(Y_syn_list[r]);
        ysyn_ptrs[r] = REAL(ysyn_holders[r]);
    }

    for (int r = 0; r < R; r++) {
        List rep = cache_list[r];
        for (int d = 0; d < 2; d++) {
            List c = rep[dir_names[d]];
            holders[r][d] = c;
            CDatL& cd = all_cd[r][d];
            cd.B = as<int>(c["B"]);
            cd.n = as<int>(c["n"]);
            cd.p = as<int>(c["p"]);
            cd.n_hon = as<int>(c["n_hon"]);
            cd.honest_idx = INTEGER(as<IntegerVector>(c["honest_idx"]));
            cd.tree_offsets = INTEGER(as<IntegerVector>(c["tree_offsets"]));
            cd.flat_sv = INTEGER(as<IntegerVector>(c["flat_sv"]));
            cd.flat_sval = REAL(as<NumericVector>(c["flat_sval"]));
            cd.flat_lc = INTEGER(as<IntegerVector>(c["flat_lc"]));
            cd.flat_rc = INTEGER(as<IntegerVector>(c["flat_rc"]));
            cd.X_row_flat = REAL(as<NumericVector>(c["X_row_flat"]));
            cd.obs_leaf_flat = INTEGER(as<IntegerVector>(c["obs_leaf_flat"]));
            cd.leaf_mean_flat = REAL(as<NumericVector>(c["leaf_mean_flat"]));
            cd.splits_on = INTEGER(as<IntegerVector>(c["splits_on"]));
            cd.fhat_obs = REAL(as<NumericVector>(c["fhat_obs"]));
        }
    }

    std::vector<double> result_flat(R * G_plus_1, NA_REAL);

    #ifdef _OPENMP
    if (n_threads > 1) omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for schedule(static) if(n_threads > 1)
    for (int r = 0; r < R; r++) {
        const double* Y_syn = ysyn_ptrs[r];
        std::vector<double> mu_accum(G_plus_1, 0.0);

        for (int d = 0; d < 2; d++) {
            const CDatL& cd = all_cd[r][d];
            int B = cd.B, n = cd.n, p = cd.p, n_hon = cd.n_hon;

            // sigma2_ej
            double ss = 0.0;
            for (int j = 0; j < n_hon; j++) {
                int i = cd.honest_idx[j] - 1;
                double ej = xv_ptr[i] - g_ptr[i];
                ss += ej * ej;
            }
            double sigma2_ej = ss / n_hon;

            // Grid accumulation: flat, tree-outer
            std::vector<double> gsum(G_plus_1 * n, 0.0);
            std::vector<int> gcnt(G_plus_1 * n, 0);

            for (int bb = 0; bb < B; bb++) {
                int off = cd.tree_offsets[bb];
                const int* sv = cd.flat_sv + off;
                const double* sval = cd.flat_sval + off;
                const int* lc = cd.flat_lc + off;
                const int* rc = cd.flat_rc + off;
                const double* lm = cd.leaf_mean_flat + off;
                int leaf_off = bb * n_hon;
                bool tree_has_split = (cd.splits_on[var_col * B + bb] != 0);

                if (tree_has_split) {
                    for (int j = 0; j < n_hon; j++) {
                        int i = cd.honest_idx[j] - 1;
                        int xbase = j * p;

                        // Shared-prefix
                        int prefix_node = 0;
                        while (lc[prefix_node] != 0 || rc[prefix_node] != 0) {
                            if (sv[prefix_node] == var_col) break;
                            prefix_node = (cd.X_row_flat[xbase + sv[prefix_node]] <= sval[prefix_node]) ?
                                          lc[prefix_node] : rc[prefix_node];
                        }
                        if (lc[prefix_node] == 0 && rc[prefix_node] == 0) {
                            if (lm[prefix_node] != -1e308) {
                                double v = lm[prefix_node];
                                for (int g = 0; g < G_plus_1; g++) {
                                    gsum[g * n + i] += v; gcnt[g * n + i]++;
                                }
                            }
                        } else {
                            for (int g = 0; g < G_plus_1; g++) {
                                double gval = grid_ptr[g];
                                int node = (gval <= sval[prefix_node]) ?
                                           lc[prefix_node] : rc[prefix_node];
                                while (lc[node] != 0 || rc[node] != 0) {
                                    double xv = (sv[node] == var_col) ? gval :
                                                cd.X_row_flat[xbase + sv[node]];
                                    node = (xv <= sval[node]) ? lc[node] : rc[node];
                                }
                                if (lm[node] != -1e308) {
                                    gsum[g * n + i] += lm[node]; gcnt[g * n + i]++;
                                }
                            }
                        }
                    }
                } else {
                    for (int j = 0; j < n_hon; j++) {
                        int i = cd.honest_idx[j] - 1;
                        int leaf = cd.obs_leaf_flat[leaf_off + j];
                        if (lm[leaf] != -1e308) {
                            double v = lm[leaf];
                            for (int g = 0; g < G_plus_1; g++) {
                                gsum[g * n + i] += v; gcnt[g * n + i]++;
                            }
                        }
                    }
                }
            }

            // AIPW at each grid point
            for (int g = 0; g < G_plus_1; g++) {
                double phi_sum = 0.0; int cnt = 0;
                for (int j = 0; j < n_hon; j++) {
                    int i = cd.honest_idx[j] - 1;
                    double fg = (gcnt[g * n + i] > 0) ? gsum[g * n + i] / gcnt[g * n + i] : NA_REAL;
                    double fo = cd.fhat_obs[j];
                    if (ISNA(fg) || ISNA(fo)) continue;
                    double omega = (xv_ptr[i] - g_ptr[i]) / sigma2_ej;
                    double R_k = Y_syn[i] - fo;
                    phi_sum += fg + omega * R_k;
                    cnt++;
                }
                mu_accum[g] += (cnt > 0) ? phi_sum / cnt : 0.0;
            }
        }

        // Average over 2 directions
        for (int g = 0; g < G_plus_1; g++)
            result_flat[r * G_plus_1 + g] = mu_accum[g] / 2.0;
    }

    NumericMatrix result(R, G_plus_1);
    for (int r = 0; r < R; r++)
        for (int g = 0; g < G_plus_1; g++)
            result(r, g) = result_flat[r * G_plus_1 + g];
    return result;
}


// ============================================================
// ranger_batch_predict_cpp
//
// Pre-extracts N ranger forests into flat arrays, then predicts
// all of them at query points in one call. Returns either:
// - For conditional: nk x N matrix of predictions (one per forest)
// - For unconditional: nk x M matrix of f_bar (averaged over K per bootstrap)
//
// forests_list: list of stripped ranger forest objects
// X_query: nk x p matrix of query points (column-major)
// group_sizes: integer vector of length M, each entry = K (forests per group)
//              If NULL, no grouping — return raw predictions
// outcome_binary: if TRUE, extract column 2 of probability predictions
// n_threads: OpenMP threads
// ============================================================
