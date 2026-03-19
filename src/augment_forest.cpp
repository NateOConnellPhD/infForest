// augment_forest.cpp — Leaf augmentation for binary predictors
//
// augment_forest_with_splits: takes a forest, returns a new forest with
// forced X_j splits added at terminal leaves where counterfactuals land
// in the same leaf.
//
// For binary X_j: the forced split at 0.5 perfectly separates X_j = 0
// and X_j = 1 groups. Within-daughter X_j variance is zero, so the
// AIPW correction introduces no inflation.
//
// For continuous X_j: augmentation is NOT used because within-daughter
// X_j variance > 0 creates structural inflation.

#include <Rcpp.h>
#include <vector>
#include <unordered_set>
using namespace Rcpp;


// [[Rcpp::export]]
List augment_forest_with_splits(
    List forest,
    NumericMatrix X_obs,
    IntegerVector honest_idx,
    int var_col,
    double cutpoint,
    double query_a,
    double query_b
) {
    List svl = forest["split.varIDs"];
    List svall = forest["split.values"];
    List chl = forest["child.nodeIDs"];
    int B = svl.size();
    int n = X_obs.nrow();
    int n_hon = honest_idx.size();

    const double* X_ptr = REAL(X_obs);

    List new_svl(B);
    List new_svall(B);
    List new_chl(B);

    for (int b = 0; b < B; b++) {
        IntegerVector sv_r = svl[b]; NumericVector sval_r = svall[b];
        List ch_orig = chl[b];
        IntegerVector lc_r = ch_orig[0], rc_r = ch_orig[1];

        std::vector<int> sv(sv_r.begin(), sv_r.end());
        std::vector<double> sval(sval_r.begin(), sval_r.end());
        std::vector<int> lc(lc_r.begin(), lc_r.end());
        std::vector<int> rc(rc_r.begin(), rc_r.end());
        int orig_size = (int)sv.size();

        std::unordered_set<int> needs_augment;

        for (int j = 0; j < n_hon; j++) {
            int i = honest_idx[j] - 1;

            int node_a = 0;
            while (lc[node_a] != 0 || rc[node_a] != 0) {
                double xval = (sv[node_a] == var_col) ? query_a : X_ptr[i + n * sv[node_a]];
                node_a = (xval <= sval[node_a]) ? lc[node_a] : rc[node_a];
            }

            int node_b = 0;
            while (lc[node_b] != 0 || rc[node_b] != 0) {
                double xval = (sv[node_b] == var_col) ? query_b : X_ptr[i + n * sv[node_b]];
                node_b = (xval <= sval[node_b]) ? lc[node_b] : rc[node_b];
            }

            if (node_a == node_b) {
                needs_augment.insert(node_a);
            }
        }

        for (int leaf : needs_augment) {
            if (leaf >= orig_size) continue;
            if (lc[leaf] != 0 || rc[leaf] != 0) continue;

            int new_left = (int)sv.size();
            int new_right = new_left + 1;

            sv[leaf] = var_col;
            sval[leaf] = cutpoint;
            lc[leaf] = new_left;
            rc[leaf] = new_right;

            sv.push_back(0); sval.push_back(0.0);
            lc.push_back(0); rc.push_back(0);
            sv.push_back(0); sval.push_back(0.0);
            lc.push_back(0); rc.push_back(0);
        }

        new_svl[b] = IntegerVector(sv.begin(), sv.end());
        new_svall[b] = NumericVector(sval.begin(), sval.end());
        List new_ch = List::create(
            IntegerVector(lc.begin(), lc.end()),
            IntegerVector(rc.begin(), rc.end())
        );
        new_chl[b] = new_ch;
    }

    List new_forest = clone(forest);
    new_forest["split.varIDs"] = new_svl;
    new_forest["split.values"] = new_svall;
    new_forest["child.nodeIDs"] = new_chl;

    return new_forest;
}
