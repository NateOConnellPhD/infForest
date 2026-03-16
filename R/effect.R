#' Estimate the Effect of a Predictor
#'
#' Computes population-averaged effects of a predictor using honest within-leaf
#' contrasts. For binary predictors, returns the average difference between
#' groups. For continuous predictors, returns per-unit slopes between all
#' pairwise combinations of analyst-specified comparison points.
#'
#' @param object An \code{infForest} object fitted with \code{honesty = TRUE}.
#' @param var Character; name of the predictor variable.
#' @param at Numeric vector of comparison points for continuous predictors.
#'   Default \code{c(0.25, 0.75)}. Interpretation depends on \code{type}.
#'   All pairwise contrasts are returned. Ignored for binary predictors.
#' @param type How to interpret \code{at}: \code{"quantile"} (default) treats
#'   values as quantile probabilities, \code{"value"} treats them as raw values.
#' @param q_lo,q_hi Quantiles defining the grid bounds for local slope
#'   estimation. Default 0.10 and 0.90.
#' @param bw Bandwidth: target number of honest observations per grid interval.
#'   Controls grid density. Default 20.
#' @param subset Optional integer vector of observation indices to restrict
#'   honest estimation to. The forest routing is unchanged; only the specified
#'   observations contribute to the honest contrasts. Useful for conditional
#'   resolution: \code{effect(fit, "x1", subset = which(dat$x3 > 0))}.
#' @param ... Additional arguments (currently unused).
#'
#' @return For binary predictors, a list of class \code{infForest_effect} with
#'   a single estimate. For continuous predictors, a list containing a data
#'   frame \code{contrasts} with all pairwise comparisons (hi, lo, hi_val,
#'   lo_val, estimate).
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#' effect(fit, "treatment")
#' effect(fit, "age")
#' effect(fit, "age", at = c(0.10, 0.50, 0.90))
#' effect(fit, "age", at = c(50, 60, 70), type = "value")
#' effect(fit, "xC", subset = which(dat$xS > quantile(dat$xS, 0.75)))
#' }
#'
#' @export
effect <- function(object, ...) UseMethod("effect")

#' @rdname effect
#' @export
effect.infForest <- function(object, var, at = c(0.25, 0.75),
                             type = c("quantile", "value"),
                             q_lo = 0.10, q_hi = 0.90,
                             bw = 20L, subset = NULL, ...) {

  check_infForest(object)
  check_varname(object, var)


  type <- match.arg(type)
  x_var <- object$X[[var]]
  var_type <- detect_var_type(x_var)

  if (var_type == "binary") {
    est <- .honest_effect_binary(object, var, subset = subset)
    out <- list(
      variable = var,
      var_type = var_type,
      estimate = est,
      n_intervals = 1L,
      subset = subset
    )
    class(out) <- "infForest_effect"
    return(out)
  }

  if (var_type != "continuous") {
    stop("Categorical predictors with >2 levels not yet supported.")
  }

  # Resolve comparison points
  if (type == "quantile") {
    at_vals <- unname(quantile(x_var, at))
    at_labels <- paste0("Q", round(at * 100))
  } else {
    at_vals <- at
    at_labels <- as.character(round(at, 3))
  }

  at_vals <- sort(at_vals)
  at_labels <- at_labels[order(at)]

  # Build the curve once, then read off all contrasts
  n_honest <- nrow(object$X) %/% 2
  n_intervals <- max(1L, as.integer(n_honest / bw))

  # Grid must span all comparison points
  grid_lo <- min(at_vals, unname(quantile(x_var, q_lo)))
  grid_hi <- max(at_vals, unname(quantile(x_var, q_hi)))

  curve_result <- .honest_build_curve(object, var, grid_lo, grid_hi,
                                       n_honest = n_honest, bw = bw,
                                       subset = subset)

  # Extract all pairwise contrasts (hi > lo)
  n_at <- length(at_vals)
  pairs <- combn(n_at, 2)
  n_pairs <- ncol(pairs)

  contrasts_df <- data.frame(
    hi = character(n_pairs),
    lo = character(n_pairs),
    hi_val = numeric(n_pairs),
    lo_val = numeric(n_pairs),
    estimate = numeric(n_pairs),
    stringsAsFactors = FALSE
  )

  # Global augmentation correction for continuous variables
  # Predict with X_j at median (reference), compare groups defined by at_vals
  global_corrections <- numeric(n_pairs)
  for (k in seq_len(n_pairs)) {
    i_lo <- pairs[1, k]
    i_hi <- pairs[2, k]
    val_lo_k <- at_vals[i_lo]
    val_hi_k <- at_vals[i_hi]
    mid_ref <- (val_lo_k + val_hi_k) / 2
    X_ref <- object$X; X_ref[[var]] <- mid_ref
    # Average predictions across all forests (both directions)
    all_pred_ref <- numeric(nrow(object$X))
    n_forests <- 0
    for (r in seq_along(object$forests)) {
      fs <- object$forests[[r]]
      all_pred_ref <- all_pred_ref + predict(fs$rfA, data = X_ref)$predictions
      all_pred_ref <- all_pred_ref + predict(fs$rfB, data = X_ref)$predictions
      n_forests <- n_forests + 2
    }
    all_pred_ref <- all_pred_ref / n_forests

    idx_hi <- x_var >= val_hi_k
    idx_lo <- x_var <= val_lo_k
    if (sum(idx_hi) > 0 && sum(idx_lo) > 0) {
      global_corrections[k] <- (mean(all_pred_ref[idx_hi]) - mean(all_pred_ref[idx_lo])) /
                                (val_hi_k - val_lo_k)
    }
  }

  for (k in seq_len(n_pairs)) {
    i_lo <- pairs[1, k]
    i_hi <- pairs[2, k]
    val_hi <- approx(curve_result$grid, curve_result$curve, xout = at_vals[i_hi], rule = 2)$y
    val_lo <- approx(curve_result$grid, curve_result$curve, xout = at_vals[i_lo], rule = 2)$y
    contrasts_df$hi[k] <- at_labels[i_hi]
    contrasts_df$lo[k] <- at_labels[i_lo]
    contrasts_df$hi_val[k] <- at_vals[i_hi]
    contrasts_df$lo_val[k] <- at_vals[i_lo]
    raw_slope <- (val_hi - val_lo) / (at_vals[i_hi] - at_vals[i_lo])
    contrasts_df$estimate[k] <- raw_slope - global_corrections[k]
  }

  out <- list(
    variable = var,
    var_type = var_type,
    contrasts = contrasts_df,
    at = at,
    at_vals = at_vals,
    at_labels = at_labels,
    type = type,
    n_intervals = n_intervals,
    subset = subset
  )
  class(out) <- "infForest_effect"
  out
}


#' @keywords internal
.honest_effect_binary <- function(object, var, subset = NULL) {
  all_estimates <- numeric(object$honesty.splits)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    hon_B <- if (!is.null(subset)) intersect(fs$idxB, subset) else fs$idxB
    hon_A <- if (!is.null(subset)) intersect(fs$idxA, subset) else fs$idxA
    est_AB <- .extract_binary_one_direction(fs$rfA, object$X, object$Y,
                                             honest_idx = hon_B, var = var)
    est_BA <- .extract_binary_one_direction(fs$rfB, object$X, object$Y,
                                             honest_idx = hon_A, var = var)
    all_estimates[r] <- (est_AB + est_BA) / 2
  }

  mean(all_estimates)
}


#' @keywords internal
.honest_build_curve <- function(object, var, grid_lo, grid_hi, n_honest, bw,
                                subset = NULL) {
  n_intervals <- max(1L, as.integer(n_honest / bw))
  grid <- seq(grid_lo, grid_hi, length.out = n_intervals + 1)

  all_slopes <- matrix(0, nrow = object$honesty.splits, ncol = n_intervals)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    hon_B <- if (!is.null(subset)) intersect(fs$idxB, subset) else fs$idxB
    hon_A <- if (!is.null(subset)) intersect(fs$idxA, subset) else fs$idxA
    slopes_AB <- .extract_curve_slopes(fs$rfA, object$X, object$Y,
                                       honest_idx = hon_B, var = var,
                                       grid = grid)
    slopes_BA <- .extract_curve_slopes(fs$rfB, object$X, object$Y,
                                       honest_idx = hon_A, var = var,
                                       grid = grid)
    all_slopes[r, ] <- (slopes_AB + slopes_BA) / 2
  }

  avg_slopes <- colMeans(all_slopes)
  intervals <- diff(grid)
  curve_vals <- c(0, cumsum(avg_slopes * intervals))

  list(grid = grid, curve = curve_vals, slopes = avg_slopes, intervals = intervals)
}


#' @keywords internal
.extract_binary_one_direction <- function(rf, X, Y, honest_idx, var) {
  X_ord <- reorder_X_to_ranger(X, rf)
  col_idx <- get_ranger_col_idx(rf, var)

  n <- nrow(X)
  y_hon <- rep(NA_real_, n)
  y_hon[honest_idx] <- as.numeric(Y[honest_idx])

  # Precompute forest-wide non-X_j prediction (X_j set to 0 for all)
  X_ref0 <- X; X_ref0[[var]] <- 0
  pred_ref <- predict(rf, data = X_ref0)$predictions

  # Raw contrast
  res <- honest_all(
    rf$forest, X_ord, y_hon, as.integer(honest_idx),
    bin_cols = as.integer(col_idx),
    cont_cols = as.integer(integer(0)),
    cont_thresh = numeric(0),
    per_leaf_denom = TRUE
  )

  raw_popavg <- res$binary$popavg[1]

  # Global augmentation: use ALL observations (imbalance is a property of X, not the fold)
  x_vals <- X[[var]]
  fref_vals <- pred_ref
  idx1 <- x_vals > 0.5
  idx0 <- !idx1
  if (sum(idx1) > 0 && sum(idx0) > 0) {
    fref_mean_1 <- mean(fref_vals[idx1])
    fref_mean_0 <- mean(fref_vals[idx0])
    global_correction <- fref_mean_1 - fref_mean_0
  } else {
    fref_mean_1 <- NA; fref_mean_0 <- NA
    global_correction <- 0
  }

  raw_popavg - global_correction
}


#' @keywords internal
.extract_binary_multi_one_direction <- function(rf, X, Y, honest_idx, vars) {
  X_ord <- reorder_X_to_ranger(X, rf)
  col_idxs <- vapply(vars, function(v) get_ranger_col_idx(rf, v), integer(1))

  n <- nrow(X)
  y_hon <- rep(NA_real_, n)
  y_hon[honest_idx] <- as.numeric(Y[honest_idx])

  # Raw contrasts (no within-leaf augmentation)
  res <- honest_all(
    rf$forest, X_ord, y_hon, as.integer(honest_idx),
    bin_cols = as.integer(col_idxs),
    cont_cols = as.integer(integer(0)),
    cont_thresh = numeric(0),
    per_leaf_denom = TRUE
  )

  raw <- res$binary$popavg

  # Global augmentation for each binary variable
  out <- numeric(length(vars))
  for (j in seq_along(vars)) {
    X_ref0 <- X; X_ref0[[vars[j]]] <- 0
    pred_ref <- predict(rf, data = X_ref0)$predictions

    x_vals <- X[[vars[j]]]
    fref_vals <- pred_ref
    idx1 <- x_vals > 0.5
    idx0 <- !idx1
    if (sum(idx1) > 0 && sum(idx0) > 0) {
      global_correction <- mean(fref_vals[idx1]) - mean(fref_vals[idx0])
    } else {
      global_correction <- 0
    }
    out[j] <- raw[j] - global_correction
  }

  names(out) <- vars
  out
}


#' @keywords internal
.honest_effect_binary_multi <- function(object, vars) {
  # Estimate multiple binary effects in one pass per forest
  all_estimates <- matrix(0, nrow = object$honesty.splits, ncol = length(vars))
  colnames(all_estimates) <- vars

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    est_AB <- .extract_binary_multi_one_direction(fs$rfA, object$X, object$Y,
                                                   honest_idx = fs$idxB, vars = vars)
    est_BA <- .extract_binary_multi_one_direction(fs$rfB, object$X, object$Y,
                                                   honest_idx = fs$idxA, vars = vars)
    all_estimates[r, ] <- (est_AB + est_BA) / 2
  }

  colMeans(all_estimates)
}



#' @keywords internal
.grid_to_windows <- function(grid) {
  G <- length(grid)
  midpts <- (grid[-1] + grid[-G]) / 2
  intervals <- diff(grid)
  wlo <- grid[-G] - intervals
  whi <- grid[-1] + intervals
  list(midpts = midpts, intervals = intervals, wlo = wlo, whi = whi)
}


#' Print method for infForest_effect objects
#'
#' @param x An \code{infForest_effect} object.
#' @param ... Additional arguments (ignored).
#' @export
print.infForest_effect <- function(x, ...) {
  cat("Inference Forest Effect Estimate\n")
  cat("  Variable:   ", x$variable, "\n")
  cat("  Type:       ", x$var_type, "\n")

  if (x$var_type == "binary") {
    cat("  Estimate:   ", round(x$estimate, 4), "\n")
    if (!is.null(x$se)) {
      cat("  SE:         ", round(x$se, 4), "\n")
      cat("  95% CI:      [", round(x$ci_lower, 4), ", ", round(x$ci_upper, 4), "]\n")
      cat("  p-value:    ", format.pval(x$pval, digits = 3), "\n")
    }
  } else {
    cat("  Intervals:  ", x$n_intervals, "\n")
    cat("\n  Pairwise contrasts (per unit):\n")
    df <- x$contrasts
    for (k in seq_len(nrow(df))) {
      cat(sprintf("    %s vs %s  [%.3f vs %.3f]:  %.4f\n",
                  df$hi[k], df$lo[k], df$hi_val[k], df$lo_val[k], df$estimate[k]))
    }
    if (!is.null(x$se)) {
      cat(sprintf("\n  Primary contrast SE: %.4f\n", x$se))
      cat(sprintf("  95%% CI: [%.4f, %.4f]\n", x$ci_lower, x$ci_upper))
      cat(sprintf("  p-value: %s\n", format.pval(x$pval, digits = 3)))
    }
  }
  invisible(x)
}
