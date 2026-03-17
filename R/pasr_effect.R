#' PASR for Effect Functionals
#'
#' Estimates the covariance floor C_Psi for any effect functional using
#' scalar-first procedure-aligned synthetic resampling. Generates synthetic
#' outcome replicates, fits paired inference forests on each, extracts the
#' same effect functional from both, and estimates C_Psi from the
#' cross-covariance of paired scalars.
#'
#' @param object An \code{infForest} object.
#' @param var Character; name of the predictor variable.
#' @param at Comparison points for continuous predictors. Default c(0.25, 0.75).
#' @param type How to interpret \code{at}. Default "quantile".
#' @param bw Bandwidth. Default 20.
#' @param q_lo,q_hi Grid bounds. Default 0.10 and 0.90.
#' @param subset Optional subset indices for conditional resolution.
#' @param R Number of synthetic replicates. Default 50.
#' @param B_mc Number of trees per paired forest in PASR. Default 500.
#' @param alpha Significance level. Default 0.05.
#' @param nuisance An \code{infForest_nuisance} object. If \code{NULL},
#'   estimated automatically.
#' @param ... Additional arguments passed to \code{estimate_nuisance}.
#'
#' @return A list of class \code{infForest_effect} with added fields:
#' \describe{
#'   \item{se}{Standard error of the effect estimate.}
#'   \item{ci_lower, ci_upper}{Confidence interval.}
#'   \item{pval}{Two-sided p-value.}
#'   \item{C_psi}{Covariance floor estimate.}
#'   \item{V_psi}{Monte Carlo variance component.}
#' }
#'
#' @export
pasr_effect <- function(object, var, at = c(0.25, 0.75),
                        type = c("quantile", "value"),
                        bw = 20L, q_lo = 0.10, q_hi = 0.90,
                        subset = NULL,
                        R_min = 20L, R_max = 200L, batch_size = 10L,
                        tol = 0.05, n_stable = 2L,
                        B_mc = 500L, alpha = 0.05,
                        nuisance = NULL, verbose = FALSE, ...) {

  check_infForest(object)
  check_varname(object, var)

  type <- match.arg(type)
  z_crit <- qnorm(1 - alpha / 2)

  X <- object$X
  Y <- object$Y
  n <- nrow(X)

  # --- Get the point estimate from the deployed forest ---
  deployed_effect <- effect(object, var, at = at, type = type,
                            bw = bw, q_lo = q_lo, q_hi = q_hi,
                            subset = subset)

  # --- Within-forest MC variance (V_psi / B) ---
  # Compute per-split estimates and take their variance
  var_type <- detect_var_type(X[[var]])
  split_estimates <- numeric(object$honesty.splits)

  for (r_split in seq_along(object$forests)) {
    fs <- object$forests[[r_split]]
    if (var_type == "binary") {
      hon_B <- if (!is.null(subset)) intersect(fs$idxB, subset) else fs$idxB
      hon_A <- if (!is.null(subset)) intersect(fs$idxA, subset) else fs$idxA
      est_AB <- .extract_binary_one_direction(fs$rfA, X, Y, honest_idx = hon_B, var = var)
      est_BA <- .extract_binary_one_direction(fs$rfB, X, Y, honest_idx = hon_A, var = var)
      split_estimates[r_split] <- (est_AB + est_BA) / 2
    } else {
      # For continuous, use the curve-based estimate
      x_var <- X[[var]]
      if (type == "quantile") {
        at_vals <- sort(unname(quantile(x_var, at)))
      } else {
        at_vals <- sort(at)
      }
      a <- at_vals[length(at_vals)]; b <- at_vals[1]
      grid_lo_val <- min(at_vals, unname(quantile(x_var, q_lo)))
      grid_hi_val <- max(at_vals, unname(quantile(x_var, q_hi)))
      n_honest <- n %/% 2
      n_intervals <- max(1L, as.integer(n_honest / bw))
      grid <- seq(grid_lo_val, grid_hi_val, length.out = n_intervals + 1)

      fs <- object$forests[[r_split]]
      hon_B <- if (!is.null(subset)) intersect(fs$idxB, subset) else fs$idxB
      hon_A <- if (!is.null(subset)) intersect(fs$idxA, subset) else fs$idxA

      slopes_AB <- .extract_curve_slopes(fs$rfA, X, Y, honest_idx = hon_B,
                                         var = var, grid = grid)
      slopes_BA <- .extract_curve_slopes(fs$rfB, X, Y, honest_idx = hon_A,
                                         var = var, grid = grid)
      avg_slopes <- (slopes_AB + slopes_BA) / 2
      intervals <- diff(grid)
      curve_vals <- c(0, cumsum(avg_slopes * intervals))
      val_a <- approx(grid, curve_vals, xout = a, rule = 2)$y
      val_b <- approx(grid, curve_vals, xout = b, rule = 2)$y
      split_estimates[r_split] <- (val_a - val_b) / (a - b)
    }
  }
  # V_psi estimated from variance across honesty splits
  # This captures both MC and fold-assignment variance
  V_psi_over_R <- var(split_estimates) / object$honesty.splits

  # --- Estimate nuisance if not provided ---
  if (is.null(nuisance)) {
    nuisance <- estimate_nuisance(object, ...)
  }

  # --- Scalar-first PASR with convergence ---
  psi_A <- numeric(0)
  psi_B <- numeric(0)
  R_current <- 0L
  C_prev <- Inf
  stable_count <- 0L
  converged <- FALSE

  while (R_current < R_max) {
    # Run one batch
    for (b in seq_len(batch_size)) {
      R_current <- R_current + 1L
      r <- R_current

      Y_syn <- generate_synthetic_Y(nuisance, seed = r * 7919L)

      dat_syn <- X
      if (object$outcome_type == "continuous") {
        dat_syn$y <- Y_syn
      } else {
        dat_syn$y <- factor(Y_syn, levels = c(0, 1))
      }

      # Shared fold assignment for paired forests
      set.seed(r * 5113L)
      shared_fold <- list(sample(rep(1:2, length.out = n)))

      fitA <- infForest(y ~ ., data = dat_syn, num.trees = B_mc,
                        mtry = object$params$mtry,
                        min.node.size = object$params$min.node.size,
                        penalize = object$params$penalize,
                        softmax = object$params$softmax,
                        honesty.splits = 1L,
                        fold_assignments = shared_fold,
                        seed = r * 100L + 1L)

      fitB <- infForest(y ~ ., data = dat_syn, num.trees = B_mc,
                        mtry = object$params$mtry,
                        min.node.size = object$params$min.node.size,
                        penalize = object$params$penalize,
                        softmax = object$params$softmax,
                        honesty.splits = 1L,
                        fold_assignments = shared_fold,
                        seed = r * 100L + 2L)

      effA <- effect(fitA, var, at = at, type = type, bw = bw,
                     q_lo = q_lo, q_hi = q_hi, subset = subset)
      effB <- effect(fitB, var, at = at, type = type, bw = bw,
                     q_lo = q_lo, q_hi = q_hi, subset = subset)

      if (var_type == "binary") {
        psi_A <- c(psi_A, effA$estimate)
        psi_B <- c(psi_B, effB$estimate)
      } else {
        psi_A <- c(psi_A, effA$contrasts$estimate[1])
        psi_B <- c(psi_B, effB$contrasts$estimate[1])
      }
    }

    # Check convergence after R_min
    if (R_current >= R_min) {
      C_current <- max(cov(psi_A, psi_B), 0)
      if (is.finite(C_prev)) {
        rel_change <- abs(C_current - C_prev) / max(C_prev, 1e-10)
      } else {
        rel_change <- Inf
      }

      if (verbose) {
        cat(sprintf("  PASR R=%d: C_psi=%.6f  rel_change=%.4f  stable=%d/%d\n",
                    R_current, C_current, rel_change, stable_count, n_stable))
      }

      if (rel_change < tol) {
        stable_count <- stable_count + 1L
      } else {
        stable_count <- 0L
      }

      if (stable_count >= n_stable) {
        converged <- TRUE
        break
      }
      C_prev <- C_current
    }
  }

  # Final C_psi estimate
  C_psi <- max(cov(psi_A, psi_B), 0)

  if (verbose) {
    if (converged) {
      cat(sprintf("  PASR converged at R=%d (C_psi=%.6f)\n", R_current, C_psi))
    } else {
      cat(sprintf("  PASR reached R_max=%d without convergence (C_psi=%.6f)\n", R_max, C_psi))
    }
  }

  # --- Total variance and CI ---
  total_var <- V_psi_over_R + C_psi
  se <- sqrt(total_var)

  if (var_type == "binary") {
    est <- deployed_effect$estimate
  } else {
    est <- deployed_effect$contrasts$estimate[1]
  }

  ci_lower <- est - z_crit * se
  ci_upper <- est + z_crit * se
  pval <- 2 * pnorm(-abs(est / se))

  # --- Attach to deployed effect object ---
  deployed_effect$se <- se
  deployed_effect$ci_lower <- ci_lower
  deployed_effect$ci_upper <- ci_upper
  deployed_effect$pval <- pval
  deployed_effect$C_psi <- C_psi
  deployed_effect$V_psi <- V_psi_over_R
  deployed_effect$total_var <- total_var
  deployed_effect$alpha <- alpha
  deployed_effect$R_used <- R_current
  deployed_effect$converged <- converged

  deployed_effect
}
