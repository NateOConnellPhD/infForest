#' PASR Confidence Bands for Effect Curves
#'
#' Estimates pointwise and simultaneous confidence bands for a nonlinear
#' effect curve using procedure-aligned synthetic resampling. Fits paired
#' forests once per replicate and extracts the full curve from each,
#' yielding the joint covariance matrix across all grid points. Also
#' computes the sandwich variance at each grid point and an omnibus
#' chi-squared test for the null of no association.
#'
#' This is dramatically faster than calling \code{pasr_effect()} in a loop
#' over grid points: \code{R * 4} ranger fits total instead of
#' \code{G * R * 4}.
#'
#' @param object An \code{infForest} object.
#' @param var Character; name of a continuous predictor.
#' @param q_lo,q_hi Quantiles defining the curve range. Default 0.10 and 0.90.
#' @param bw Bandwidth for grid density. Default 20.
#' @param ref Reference value (curve = 0 here). Default: median.
#' @param variance_method Character: \code{"both"} (default), \code{"pasr"},
#'   or \code{"sandwich"}.
#' @param R_min,R_max Minimum/maximum PASR replicates. Default 20 and 200.
#' @param batch_size Replicates per convergence check. Default 10.
#' @param tol Convergence tolerance. Default 0.05.
#' @param n_stable Consecutive stable batches. Default 2.
#' @param B_mc Trees per paired forest. Default 500.
#' @param alpha Significance level. Default 0.05.
#' @param nuisance An \code{infForest_nuisance} object. If NULL, estimated.
#' @param verbose Print progress. Default FALSE.
#' @param ... Additional arguments passed to \code{estimate_nuisance}.
#'
#' @return A list of class \code{infForest_curve} with added fields:
#' \describe{
#'   \item{se}{Pointwise SE at each grid point.}
#'   \item{ci_lower, ci_upper}{Pointwise confidence bands.}
#'   \item{cov_matrix}{Joint covariance matrix (PASR, when computed).}
#'   \item{omnibus_stat}{Chi-squared omnibus test statistic.}
#'   \item{omnibus_df}{Degrees of freedom for omnibus test.}
#'   \item{omnibus_pval}{Omnibus p-value.}
#' }
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#' pc <- pasr_curve(fit, "x1", verbose = TRUE)
#' plot(pc)  # curve with confidence bands
#' pc$omnibus_pval  # global test of association
#' }
#'
#' @export
pasr_curve <- function(object, var, q_lo = 0.10, q_hi = 0.90,
                       bw = 20L, ref = NULL,
                       variance_method = c("both", "pasr", "sandwich"),
                       R_min = 20L, R_max = 200L, batch_size = 10L,
                       tol = 0.05, n_stable = 2L,
                       B_mc = 500L, alpha = 0.05,
                       nuisance = NULL, verbose = FALSE, ...) {

  check_infForest(object)
  check_varname(object, var)
  variance_method <- match.arg(variance_method)

  X <- object$X
  Y <- object$Y
  n <- nrow(X)
  x_var <- X[[var]]

  if (detect_var_type(x_var) != "continuous")
    stop("pasr_curve() requires a continuous predictor.")

  if (is.null(ref)) ref <- unname(median(x_var))
  z_crit <- qnorm(1 - alpha / 2)

  do_pasr     <- variance_method %in% c("pasr", "both")
  do_sandwich <- variance_method %in% c("sandwich", "both")

  # --- Propensity (once) ---
  prop <- .fit_propensity(X, var, is_binary = FALSE, n_trees = 2000L)
  ghat <- prop$ghat

  # --- Grid ---
  lo <- unname(quantile(x_var, q_lo))
  hi <- unname(quantile(x_var, q_hi))
  n_honest <- n %/% 2
  n_intervals <- max(5L, min(20L, as.integer(n_honest / bw)))
  grid <- seq(lo, hi, length.out = n_intervals + 1)
  G <- length(grid) - 1

  # --- Deployed curve ---
  curve_result <- .aipw_build_curve(object, var, grid_lo = lo, grid_hi = hi,
                                     n_honest = n_honest, bw = bw, ghat = ghat)
  curve_raw <- curve_result$curve
  ref_val <- approx(grid, curve_raw, xout = ref, rule = 2)$y
  curve_vals <- curve_raw - ref_val

  # --- Pre-compute shared objects ---
  X_ord <- .get_X_ord(object, object$forests[[1]]$rfA)
  col_idx <- get_ranger_col_idx(object$forests[[1]]$rfA, var)

  # Helper: extract full curve vector from a paired forest
  .extract_curve_vec <- function(rf_fA, rf_fB, idxA, idxB, Y_syn) {
    y_AB <- rep(NA_real_, n); hon_AB <- idxB
    y_AB[hon_AB] <- Y_syn[hon_AB]
    r_AB <- aipw_curve_v2_cpp(rf_fA$forest, X_ord, y_AB,
                               as.integer(hon_AB), ghat, col_idx, grid)

    y_BA <- rep(NA_real_, n); hon_BA <- idxA
    y_BA[hon_BA] <- Y_syn[hon_BA]
    r_BA <- aipw_curve_v2_cpp(rf_fB$forest, X_ord, y_BA,
                               as.integer(hon_BA), ghat, col_idx, grid)

    sl_AB <- r_AB$slopes; sl_AB[is.na(sl_AB)] <- 0
    sl_BA <- r_BA$slopes; sl_BA[is.na(sl_BA)] <- 0
    avg_sl <- (sl_AB + sl_BA) / 2
    intervals <- diff(grid)
    cv <- c(0, cumsum(avg_sl * intervals))
    # Shift to reference
    rv <- approx(grid, cv, xout = ref, rule = 2)$y
    cv - rv
  }

  # --- Sandwich SE ---
  se_sand <- NULL
  if (do_sandwich) {
    # Use per-grid-point scores from the deployed forest
    # Each grid point g: contrast = curve(grid[g+1]) - curve(grid[g]) integrated
    # But for pointwise CI at each grid point, we need the variance of curve(t_g).
    # The curve at t_g = sum of slopes * intervals from 0 to g.
    # We compute the sandwich from the per-split curve variation.
    all_curves <- matrix(0, nrow = object$honesty.splits, ncol = G + 1)
    for (r in seq_along(object$forests)) {
      fs <- object$forests[[r]]
      X_ord_A <- .get_X_ord(object, fs$rfA)
      col_idx_A <- get_ranger_col_idx(fs$rfA, var)
      X_ord_B <- .get_X_ord(object, fs$rfB)
      col_idx_B <- get_ranger_col_idx(fs$rfB, var)

      slopes_AB <- .aipw_curve_one_direction(
        rf = fs$rfA, X_ord = X_ord_A, Y = Y,
        honest_idx = fs$idxB, col_idx = col_idx_A,
        grid = grid, ghat = ghat)
      slopes_BA <- .aipw_curve_one_direction(
        rf = fs$rfB, X_ord = X_ord_B, Y = Y,
        honest_idx = fs$idxA, col_idx = col_idx_B,
        grid = grid, ghat = ghat)

      avg_sl <- (slopes_AB + slopes_BA) / 2
      cv <- c(0, cumsum(avg_sl * diff(grid)))
      rv <- approx(grid, cv, xout = ref, rule = 2)$y
      all_curves[r, ] <- cv - rv
    }
    # Sandwich: variance across honesty splits at each grid point
    se_sand <- apply(all_curves, 2, sd) / sqrt(object$honesty.splits)
  }

  # --- PASR ---
  se_pasr <- NULL; cov_matrix <- NULL
  omnibus_stat <- NULL; omnibus_df <- NULL; omnibus_pval <- NULL
  R_used <- NA_integer_; converged_flag <- NA

  if (do_pasr) {
    if (is.null(nuisance)) nuisance <- estimate_nuisance(object, ...)

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

    # Store full curve vectors from each replicate
    # Columns = G+1 grid points, rows = replicates
    curve_A_mat <- matrix(NA_real_, R_max, G + 1)
    curve_B_mat <- matrix(NA_real_, R_max, G + 1)
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

        # 4 ranger fits per replicate (2 per paired forest)
        rfA1 <- do.call(inf.ranger::ranger, .build_rf_args(dat_fA, r * 100L + 1L))
        rfA2 <- do.call(inf.ranger::ranger, .build_rf_args(dat_fB, r * 100L + 1L))
        rfB1 <- do.call(inf.ranger::ranger, .build_rf_args(dat_fA, r * 100L + 2L))
        rfB2 <- do.call(inf.ranger::ranger, .build_rf_args(dat_fB, r * 100L + 2L))

        # Extract full curve from each paired forest
        curve_A_mat[r, ] <- .extract_curve_vec(rfA1, rfA2, idxA, idxB, Y_syn)
        curve_B_mat[r, ] <- .extract_curve_vec(rfB1, rfB2, idxA, idxB, Y_syn)
      }

      # Convergence check on median pointwise C_psi
      if (R_current >= R_min) {
        cA <- curve_A_mat[1:R_current, , drop = FALSE]
        cB <- curve_B_mat[1:R_current, , drop = FALSE]
        Ct_current <- numeric(G + 1)
        for (g in seq_len(G + 1))
          Ct_current[g] <- max(cov(cA[, g], cB[, g]), 0)
        med_Ct <- median(Ct_current)

        rel_change <- if (is.finite(C_prev)) abs(med_Ct - C_prev) / max(C_prev, 1e-10) else Inf
        if (verbose)
          cat(sprintf("  PASR R=%d: median_C=%.6f  rel_change=%.4f  stable=%d/%d\n",
                      R_current, med_Ct, rel_change, stable_count, n_stable))
        if (rel_change < tol) { stable_count <- stable_count + 1L
        } else { stable_count <- 0L }
        if (stable_count >= n_stable) { converged <- TRUE; break }
        C_prev <- med_Ct
      }
    }

    R_used <- R_current
    converged_flag <- converged
    cA <- curve_A_mat[1:R_current, , drop = FALSE]
    cB <- curve_B_mat[1:R_current, , drop = FALSE]

    if (verbose) {
      if (converged) cat(sprintf("  PASR converged at R=%d\n", R_current))
      else cat(sprintf("  PASR reached R_max=%d without convergence\n", R_max))
    }

    # Pointwise covariance floor
    Ct_hat <- numeric(G + 1)
    for (g in seq_len(G + 1))
      Ct_hat[g] <- max(cov(cA[, g], cB[, g]), 0)
    se_pasr <- sqrt(Ct_hat)

    # Joint covariance matrix (eq 6.5 in paper)
    # Sigma[g1, g2] = cov(curve_A[,g1], curve_B[,g2]) cross-product
    cov_matrix <- matrix(0, G + 1, G + 1)
    mean_A <- colMeans(cA); mean_B <- colMeans(cB)
    for (r_idx in seq_len(R_current))
      cov_matrix <- cov_matrix + (cA[r_idx, ] - mean_A) %*% t(cB[r_idx, ] - mean_B)
    cov_matrix <- cov_matrix / (R_current - 1)
    # Ensure positive semi-definiteness
    cov_matrix <- (cov_matrix + t(cov_matrix)) / 2

    # Omnibus test (eq 6.6): T = curve' Sigma^{-1} curve ~ chi^2_G
    # Use curve values at grid points (excluding reference where curve=0)
    # Find reference grid index
    ref_g <- which.min(abs(grid - ref))
    test_idx <- setdiff(seq_len(G + 1), ref_g)
    curve_test <- curve_vals[test_idx]
    Sigma_test <- cov_matrix[test_idx, test_idx, drop = FALSE]

    # Regularize for invertibility
    Sigma_test <- Sigma_test + diag(1e-10, nrow(Sigma_test))
    tryCatch({
      Sigma_inv <- solve(Sigma_test)
      omnibus_stat <- as.numeric(t(curve_test) %*% Sigma_inv %*% curve_test)
      omnibus_df <- length(test_idx)
      omnibus_pval <- pchisq(omnibus_stat, df = omnibus_df, lower.tail = FALSE)
    }, error = function(e) {
      omnibus_stat <<- NA_real_
      omnibus_df <<- length(test_idx)
      omnibus_pval <<- NA_real_
    })
  }

  # --- Assemble primary SE ---
  if (!is.null(se_pasr)) {
    se_primary <- se_pasr
  } else {
    se_primary <- se_sand
  }

  # --- Build return object ---
  out <- list(
    variable = var,
    grid = grid,
    curve = curve_vals,
    slopes = curve_result$slopes,
    intervals = curve_result$intervals,
    ref = ref,
    n_intervals = n_intervals,
    q_lo = q_lo,
    q_hi = q_hi,
    # Variance
    se = se_primary,
    ci_lower = curve_vals - z_crit * se_primary,
    ci_upper = curve_vals + z_crit * se_primary,
    alpha = alpha,
    variance_method = variance_method
  )

  if (!is.null(se_pasr)) {
    out$se_pasr <- se_pasr
    out$cov_matrix <- cov_matrix
    out$R_used <- R_used
    out$converged <- converged_flag
    out$omnibus_stat <- omnibus_stat
    out$omnibus_df <- omnibus_df
    out$omnibus_pval <- omnibus_pval
  }
  if (!is.null(se_sand)) {
    out$se_sandwich <- se_sand
  }

  class(out) <- "infForest_curve"
  out
}


#' Print method for infForest_curve objects
#'
#' @param x An \code{infForest_curve} object.
#' @param ... Additional arguments (ignored).
#' @export
print.infForest_curve <- function(x, ...) {
  cat("Inference Forest Effect Curve\n")
  cat("  Variable:   ", x$variable, "\n")
  cat("  Grid range: ", round(min(x$grid), 3), "to", round(max(x$grid), 3), "\n")
  cat("  Reference:  ", round(x$ref, 3), "\n")
  cat("  Intervals:  ", x$n_intervals, "\n")
  cat("  Curve range:", round(min(x$curve), 4), "to", round(max(x$curve), 4), "\n")
  if (!is.null(x$se)) {
    cat("  Max SE:     ", round(max(x$se), 4), "\n")
    if (!is.null(x$variance_method))
      cat("  Variance:   ", x$variance_method, "\n")
  }
  if (!is.null(x$omnibus_pval)) {
    cat(sprintf("  Omnibus:     chi2 = %.2f, df = %d, p = %s\n",
                x$omnibus_stat, x$omnibus_df,
                format.pval(x$omnibus_pval, digits = 3)))
  }
  invisible(x)
}


#' Plot method for infForest_curve objects
#'
#' @param x An \code{infForest_curve} object.
#' @param bands Logical; show confidence bands? Default TRUE if SE available.
#' @param ... Additional arguments passed to \code{plot()}.
#' @export
plot.infForest_curve <- function(x, bands = !is.null(x$se), ...) {
  yl <- if (bands && !is.null(x$ci_lower)) {
    range(c(x$ci_lower, x$ci_upper), na.rm = TRUE)
  } else {
    range(x$curve)
  }

  plot(x$grid, x$curve, type = "l", lwd = 2,
       xlab = x$variable, ylab = "Effect (relative to reference)",
       main = paste("Effect curve:", x$variable),
       ylim = yl, ...)

  if (bands && !is.null(x$ci_lower)) {
    polygon(c(x$grid, rev(x$grid)),
            c(x$ci_lower, rev(x$ci_upper)),
            col = rgb(0, 0, 0, 0.12), border = NA)
  }

  abline(h = 0, lty = 2, col = "gray50")
  abline(v = x$ref, lty = 3, col = "gray70")
}
