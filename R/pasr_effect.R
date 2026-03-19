#' Variance Estimation and Confidence Intervals for Effect Functionals
#'
#' Estimates the standard error of any effect functional and constructs
#' confidence intervals. Three variance estimation methods are available:
#' \code{"sandwich"} (instant, from AIPW influence function scores),
#' \code{"pasr"} (procedure-aligned synthetic resampling, more thorough),
#' or \code{"both"} (computes both and reports the diagnostic ratio).
#'
#' @param object An \code{infForest} object.
#' @param var Character; name of the predictor variable.
#' @param at Comparison points for continuous predictors. Default c(0.25, 0.75).
#' @param type How to interpret \code{at}. Default "quantile".
#' @param bw Bandwidth. Default 20.
#' @param q_lo,q_hi Grid bounds. Default 0.10 and 0.90.
#' @param subset Optional subset indices for conditional resolution.
#' @param variance_method Character: \code{"sandwich"}, \code{"pasr"}, or
#'   \code{"both"} (default). The sandwich estimator is computed from AIPW
#'   influence function scores with negligible additional cost. PASR fits
#'   paired forests on synthetic outcomes to estimate the covariance floor.
#' @param R_min,R_max Minimum and maximum PASR replicates. Default 20 and 200.
#'   Ignored when \code{variance_method = "sandwich"}.
#' @param batch_size PASR replicates per convergence check. Default 10.
#' @param tol Relative change tolerance for PASR convergence. Default 0.05.
#' @param n_stable Consecutive stable batches required. Default 2.
#' @param B_mc Number of trees per paired forest in PASR. Default 500.
#' @param alpha Significance level. Default 0.05.
#' @param nuisance An \code{infForest_nuisance} object for PASR. If \code{NULL},
#'   estimated automatically. Ignored when \code{variance_method = "sandwich"}.
#' @param verbose Print PASR convergence progress. Default \code{FALSE}.
#' @param ... Additional arguments passed to \code{estimate_nuisance}.
#'
#' @return A list of class \code{infForest_effect} with added fields:
#' \describe{
#'   \item{se}{Primary standard error (from the selected method).}
#'   \item{ci_lower, ci_upper}{Confidence interval.}
#'   \item{pval}{Two-sided p-value.}
#'   \item{variance_method}{Which method was used.}
#'   \item{se_pasr}{PASR SE (when computed).}
#'   \item{se_sandwich}{Sandwich SE (when computed).}
#'   \item{C_psi}{Covariance floor estimate (PASR only).}
#'   \item{V_psi}{Monte Carlo variance component (PASR only).}
#'   \item{rho_V}{Diagnostic ratio sandwich/PASR (when both computed).}
#' }
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#'
#' # Instant SE (sandwich only, ~seconds):
#' pasr_effect(fit, "trt", variance_method = "sandwich")
#'
#' # Full PASR only:
#' pasr_effect(fit, "trt", variance_method = "pasr", verbose = TRUE)
#'
#' # Both (default): primary SE from PASR, sandwich for comparison
#' pasr_effect(fit, "trt", verbose = TRUE)
#' }
#'
#' @export
pasr_effect <- function(object, var, at = c(0.25, 0.75),
                        type = c("quantile", "value"),
                        bw = 20L, q_lo = 0.10, q_hi = 0.90,
                        subset = NULL,
                        variance_method = c("both", "sandwich", "pasr"),
                        R_min = 20L, R_max = 200L, batch_size = 10L,
                        tol = 0.05, n_stable = 2L,
                        B_mc = 500L, alpha = 0.05,
                        nuisance = NULL, verbose = FALSE, ...) {

  check_infForest(object)
  check_varname(object, var)

  type <- match.arg(type)
  variance_method <- match.arg(variance_method)
  z_crit <- qnorm(1 - alpha / 2)

  X <- object$X
  Y <- object$Y
  n <- nrow(X)

  do_pasr     <- variance_method %in% c("pasr", "both")
  do_sandwich <- variance_method %in% c("sandwich", "both")

  # --- Pre-compute propensity ONCE ---
  var_type <- detect_var_type(X[[var]])
  is_bin <- (var_type == "binary")
  prop <- .fit_propensity(X, var, is_binary = is_bin, n_trees = 2000L)
  ghat <- prop$ghat

  # --- Point estimate from the deployed forest ---
  deployed_effect <- effect(object, var, at = at, type = type,
                            bw = bw, q_lo = q_lo, q_hi = q_hi,
                            subset = subset, ghat = ghat)

  if (var_type == "binary") {
    est <- deployed_effect$estimate
  } else {
    est <- deployed_effect$contrasts$estimate[1]
  }

  # --- Sandwich SE (Section 6.4) ---
  se_sand <- NULL
  if (do_sandwich) {
    sand <- .compute_sandwich_se(object, var, at = at, type = type,
                                 bw = bw, q_lo = q_lo, q_hi = q_hi,
                                 subset = subset, ghat = ghat)
    se_sand <- sand$se
  }

  # --- PASR ---
  se_pasr <- NULL; C_psi <- NULL; V_psi_over_R <- NULL
  R_used <- NA_integer_; converged_flag <- NA

  if (do_pasr) {

    # Within-forest MC variance (V_psi / R_splits)
    split_estimates <- numeric(object$honesty.splits)
    for (r_split in seq_along(object$forests)) {
      fs <- object$forests[[r_split]]
      if (is_bin) {
        hon_B <- if (!is.null(subset)) intersect(fs$idxB, subset) else fs$idxB
        hon_A <- if (!is.null(subset)) intersect(fs$idxA, subset) else fs$idxA
        est_AB <- .extract_binary_one_direction(fs$rfA, X, Y, honest_idx = hon_B,
                                                 var = var, ghat = ghat, object = object)
        est_BA <- .extract_binary_one_direction(fs$rfB, X, Y, honest_idx = hon_A,
                                                 var = var, ghat = ghat, object = object)
        split_estimates[r_split] <- (est_AB + est_BA) / 2
      } else {
        x_var <- X[[var]]
        if (type == "quantile") { at_vals <- sort(unname(quantile(x_var, at)))
        } else { at_vals <- sort(at) }
        a <- at_vals[length(at_vals)]; b_val <- at_vals[1]
        grid_lo_val <- min(at_vals, unname(quantile(x_var, q_lo)))
        grid_hi_val <- max(at_vals, unname(quantile(x_var, q_hi)))
        n_honest <- n %/% 2
        n_intervals <- max(1L, as.integer(n_honest / bw))
        grid <- seq(grid_lo_val, grid_hi_val, length.out = n_intervals + 1)
        hon_B <- if (!is.null(subset)) intersect(fs$idxB, subset) else fs$idxB
        hon_A <- if (!is.null(subset)) intersect(fs$idxA, subset) else fs$idxA
        slopes_AB <- .extract_curve_slopes(fs$rfA, X, Y, honest_idx = hon_B,
                                            var = var, grid = grid, ghat = ghat, object = object)
        slopes_BA <- .extract_curve_slopes(fs$rfB, X, Y, honest_idx = hon_A,
                                            var = var, grid = grid, ghat = ghat, object = object)
        avg_slopes <- (slopes_AB + slopes_BA) / 2
        intervals <- diff(grid)
        curve_vals <- c(0, cumsum(avg_slopes * intervals))
        val_a <- approx(grid, curve_vals, xout = a, rule = 2)$y
        val_b <- approx(grid, curve_vals, xout = b_val, rule = 2)$y
        split_estimates[r_split] <- (val_a - val_b) / (a - b_val)
      }
    }
    V_psi_over_R <- var(split_estimates) / object$honesty.splits

    # Nuisance model
    if (is.null(nuisance)) nuisance <- estimate_nuisance(object, ...)

    # Pre-compute shared objects for PASR loop
    X_ord <- .get_X_ord(object, object$forests[[1]]$rfA)
    col_idx <- get_ranger_col_idx(object$forests[[1]]$rfA, var)

    cont_grid <- NULL; cont_a <- NULL; cont_b <- NULL
    if (!is_bin) {
      x_var <- X[[var]]
      if (type == "quantile") { at_vals <- sort(unname(quantile(x_var, at)))
      } else { at_vals <- sort(at) }
      cont_a <- at_vals[length(at_vals)]; cont_b <- at_vals[1]
      grid_lo_val <- min(at_vals, unname(quantile(x_var, q_lo)))
      grid_hi_val <- max(at_vals, unname(quantile(x_var, q_hi)))
      n_honest <- n %/% 2
      n_intervals <- max(1L, as.integer(n_honest / bw))
      cont_grid <- seq(grid_lo_val, grid_hi_val, length.out = n_intervals + 1)
    }

    .build_rf_args <- function(dat, seed_val) {
      args <- list(formula = y ~ ., data = dat, num.trees = B_mc,
                   mtry = object$params$mtry, min.node.size = object$params$min.node.size,
                   sample.fraction = 1.0, replace = FALSE, num.threads = 1L,
                   write.forest = TRUE, seed = seed_val,
                   penalize.split.competition = object$params$penalize,
                   softmax.split = object$params$softmax)
      if (object$outcome_type == "binary") args$probability <- TRUE
      args
    }

    .pasr_extract <- function(rf_fA, rf_fB, idxA, idxB, Y_syn) {
      if (is_bin) {
        y_AB <- rep(NA_real_, n); hon_AB <- if (!is.null(subset)) intersect(idxB, subset) else idxB
        y_AB[hon_AB] <- Y_syn[hon_AB]
        r_AB <- aipw_scores_v2_cpp(rf_fA$forest, X_ord, y_AB, as.integer(hon_AB), ghat, col_idx, TRUE, 1, 0)
        y_BA <- rep(NA_real_, n); hon_BA <- if (!is.null(subset)) intersect(idxA, subset) else idxA
        y_BA[hon_BA] <- Y_syn[hon_BA]
        r_BA <- aipw_scores_v2_cpp(rf_fB$forest, X_ord, y_BA, as.integer(hon_BA), ghat, col_idx, TRUE, 1, 0)
        return((r_AB$psi + r_BA$psi) / 2)
      } else {
        y_AB <- rep(NA_real_, n); hon_AB <- if (!is.null(subset)) intersect(idxB, subset) else idxB
        y_AB[hon_AB] <- Y_syn[hon_AB]
        r_AB <- aipw_curve_v2_cpp(rf_fA$forest, X_ord, y_AB, as.integer(hon_AB), ghat, col_idx, cont_grid)
        y_BA <- rep(NA_real_, n); hon_BA <- if (!is.null(subset)) intersect(idxA, subset) else idxA
        y_BA[hon_BA] <- Y_syn[hon_BA]
        r_BA <- aipw_curve_v2_cpp(rf_fB$forest, X_ord, y_BA, as.integer(hon_BA), ghat, col_idx, cont_grid)
        sl_AB <- r_AB$slopes; sl_AB[is.na(sl_AB)] <- 0
        sl_BA <- r_BA$slopes; sl_BA[is.na(sl_BA)] <- 0
        avg_sl <- (sl_AB + sl_BA) / 2
        cv <- c(0, cumsum(avg_sl * diff(cont_grid)))
        va <- approx(cont_grid, cv, xout = cont_a, rule = 2)$y
        vb <- approx(cont_grid, cv, xout = cont_b, rule = 2)$y
        return((va - vb) / (cont_a - cont_b))
      }
    }

    # PASR loop
    psi_A <- numeric(R_max); psi_B <- numeric(R_max)
    R_current <- 0L; C_prev <- Inf; stable_count <- 0L; converged <- FALSE

    while (R_current < R_max) {
      for (bi in seq_len(batch_size)) {
        R_current <- R_current + 1L; r <- R_current
        Y_syn <- generate_synthetic_Y(nuisance, seed = r * 7919L)
        dat_syn <- X
        if (object$outcome_type == "continuous") { dat_syn$y <- Y_syn
        } else { dat_syn$y <- factor(Y_syn, levels = c(0, 1)) }
        set.seed(r * 5113L)
        fold <- sample(rep(1:2, length.out = n))
        idxA <- which(fold == 1); idxB <- which(fold == 2)
        dat_fA <- dat_syn[idxA, , drop = FALSE]
        dat_fB <- dat_syn[idxB, , drop = FALSE]
        rfA1 <- do.call(inf.ranger::ranger, .build_rf_args(dat_fA, r * 100L + 1L))
        rfA2 <- do.call(inf.ranger::ranger, .build_rf_args(dat_fB, r * 100L + 1L))
        rfB1 <- do.call(inf.ranger::ranger, .build_rf_args(dat_fA, r * 100L + 2L))
        rfB2 <- do.call(inf.ranger::ranger, .build_rf_args(dat_fB, r * 100L + 2L))
        psi_A[r] <- .pasr_extract(rfA1, rfA2, idxA, idxB, Y_syn)
        psi_B[r] <- .pasr_extract(rfB1, rfB2, idxA, idxB, Y_syn)
      }
      if (R_current >= R_min) {
        pA <- psi_A[1:R_current]; pB <- psi_B[1:R_current]
        C_current <- max(cov(pA, pB), 0)
        rel_change <- if (is.finite(C_prev)) abs(C_current - C_prev) / max(C_prev, 1e-10) else Inf
        if (verbose)
          cat(sprintf("  PASR R=%d: C_psi=%.6f  rel_change=%.4f  stable=%d/%d\n",
                      R_current, C_current, rel_change, stable_count, n_stable))
        if (rel_change < tol) { stable_count <- stable_count + 1L } else { stable_count <- 0L }
        if (stable_count >= n_stable) { converged <- TRUE; break }
        C_prev <- C_current
      }
    }

    pA <- psi_A[1:R_current]; pB <- psi_B[1:R_current]
    C_psi <- max(cov(pA, pB), 0)
    if (verbose) {
      if (converged) cat(sprintf("  PASR converged at R=%d (C_psi=%.6f)\n", R_current, C_psi))
      else cat(sprintf("  PASR reached R_max=%d without convergence (C_psi=%.6f)\n", R_max, C_psi))
    }
    total_var_pasr <- V_psi_over_R + C_psi
    se_pasr <- sqrt(total_var_pasr)
    R_used <- R_current
    converged_flag <- converged
  }

  # --- Assemble results ---
  # Primary SE: PASR if available, else sandwich
  if (!is.null(se_pasr)) {
    se_primary <- se_pasr
  } else {
    se_primary <- se_sand
  }

  deployed_effect$se <- se_primary
  deployed_effect$ci_lower <- est - z_crit * se_primary
  deployed_effect$ci_upper <- est + z_crit * se_primary
  deployed_effect$pval <- 2 * pnorm(-abs(est / se_primary))
  deployed_effect$alpha <- alpha
  deployed_effect$variance_method <- variance_method

  if (!is.null(se_pasr)) {
    deployed_effect$se_pasr <- se_pasr
    deployed_effect$C_psi <- C_psi
    deployed_effect$V_psi <- V_psi_over_R
    deployed_effect$total_var <- total_var_pasr
    deployed_effect$R_used <- R_used
    deployed_effect$converged <- converged_flag
  }
  if (!is.null(se_sand)) {
    deployed_effect$se_sandwich <- se_sand
  }
  if (!is.null(se_pasr) && !is.null(se_sand)) {
    deployed_effect$rho_V <- se_sand^2 / total_var_pasr
  }

  deployed_effect
}


# ============================================================
# Sandwich (influence function) variance estimator
# ============================================================

#' @keywords internal
.compute_sandwich_se <- function(object, var, at = c(0.25, 0.75),
                                  type = "quantile", bw = 20L,
                                  q_lo = 0.10, q_hi = 0.90,
                                  subset = NULL, ghat = NULL) {

  X <- object$X
  Y <- object$Y
  n <- nrow(X)
  var_type <- detect_var_type(X[[var]])
  is_bin <- (var_type == "binary")

  if (is.null(ghat)) {
    prop <- .fit_propensity(X, var, is_binary = is_bin)
    ghat <- prop$ghat
  }

  # Collect per-observation phi scores across all honesty splits.
  # Each obs contributes to one fold direction per split.
  phi_sum <- numeric(n)
  phi_cnt <- integer(n)

  # Resolve query values for continuous
  if (!is_bin) {
    x_var <- X[[var]]
    if (type == "quantile") { at_vals <- sort(unname(quantile(x_var, at)))
    } else { at_vals <- sort(at) }
    a_val <- at_vals[length(at_vals)]
    b_val <- at_vals[1]
  }

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    X_ord_A <- .get_X_ord(object, fs$rfA)
    col_idx_A <- get_ranger_col_idx(fs$rfA, var)
    X_ord_B <- .get_X_ord(object, fs$rfB)
    col_idx_B <- get_ranger_col_idx(fs$rfB, var)

    hon_B <- if (!is.null(subset)) intersect(fs$idxB, subset) else fs$idxB
    hon_A <- if (!is.null(subset)) intersect(fs$idxA, subset) else fs$idxA

    if (is_bin) {
      a_sc <- 1; b_sc <- 0
    } else {
      a_sc <- a_val; b_sc <- b_val
    }

    # A -> B direction
    y_hon_AB <- rep(NA_real_, n); y_hon_AB[hon_B] <- as.numeric(Y[hon_B])
    res_AB <- aipw_scores_v2_cpp(fs$rfA$forest, X_ord_A, y_hon_AB,
                                  as.integer(hon_B), ghat,
                                  col_idx_A, is_bin, a_sc, b_sc)
    for (j in seq_along(hon_B)) {
      k <- hon_B[j]
      if (!is.na(res_AB$phi[j])) { phi_sum[k] <- phi_sum[k] + res_AB$phi[j]; phi_cnt[k] <- phi_cnt[k] + 1L }
    }

    # B -> A direction
    y_hon_BA <- rep(NA_real_, n); y_hon_BA[hon_A] <- as.numeric(Y[hon_A])
    res_BA <- aipw_scores_v2_cpp(fs$rfB$forest, X_ord_B, y_hon_BA,
                                  as.integer(hon_A), ghat,
                                  col_idx_B, is_bin, a_sc, b_sc)
    for (j in seq_along(hon_A)) {
      k <- hon_A[j]
      if (!is.na(res_BA$phi[j])) { phi_sum[k] <- phi_sum[k] + res_BA$phi[j]; phi_cnt[k] <- phi_cnt[k] + 1L }
    }
  }

  # Average phi per observation, compute sandwich variance
  valid <- phi_cnt > 0
  n_valid <- sum(valid)
  phi_avg <- rep(NA_real_, n)
  phi_avg[valid] <- phi_sum[valid] / phi_cnt[valid]

  psi_bar <- mean(phi_avg[valid])
  V_IF <- sum((phi_avg[valid] - psi_bar)^2) / (n_valid * (n_valid - 1))

  list(se = sqrt(V_IF), V_IF = V_IF, n_valid = n_valid, psi_bar = psi_bar)
}
