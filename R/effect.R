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

    # FWL denominator correction: compute e_j = X_j - g_hat(X_{-j})
    # Average g_hat across all forests
    X_minus_j <- object$X[, setdiff(names(object$X), var), drop = FALSE]
    g_hat_avg <- numeric(nrow(object$X))
    n_g <- 0
    for (r in seq_along(object$forests)) {
      fs <- object$forests[[r]]
      for (rf_build_idx in list(fs$idxA, fs$idxB)) {
        dat_g <- X_minus_j[rf_build_idx, , drop = FALSE]
        dat_g$xj <- x_var[rf_build_idx]
        rf_g <- ranger::ranger(xj ~ ., data = dat_g, num.trees = 500,
                                mtry = min(5L, ncol(dat_g) - 1),
                                min.node.size = 5, seed = 43)
        g_hat_avg <- g_hat_avg + predict(rf_g, data = X_minus_j)$predictions
        n_g <- n_g + 1
      }
    }
    g_hat_avg <- g_hat_avg / n_g
    e_j <- x_var - g_hat_avg

    idx_hi <- x_var >= at_vals[i_hi]
    idx_lo <- x_var <= at_vals[i_lo]
    if (sum(idx_hi) > 0 && sum(idx_lo) > 0) {
      denom_ratio <- (mean(e_j[idx_hi]) - mean(e_j[idx_lo])) / (at_vals[i_hi] - at_vals[i_lo])
      if (abs(denom_ratio) > 0.05) {
        contrasts_df$estimate[k] <- raw_slope / denom_ratio
      } else {
        contrasts_df$estimate[k] <- raw_slope
      }
    } else {
      contrasts_df$estimate[k] <- raw_slope
    }
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

    fwl_AB <- .residualize_FWL(object$X, object$Y, fs$idxA, hon_B, var)
    fwl_BA <- .residualize_FWL(object$X, object$Y, fs$idxB, hon_A, var)

    # Numerator: honest contrast on e_Y
    raw_AB <- .extract_binary_one_direction(fs$rfA, object$X, fwl_AB$Y_resid,
                                             honest_idx = hon_B, var = var)
    raw_BA <- .extract_binary_one_direction(fs$rfB, object$X, fwl_BA$Y_resid,
                                             honest_idx = hon_A, var = var)

    # Denominator correction: mean(e_j | x_j=1) - mean(e_j | x_j=0)
    x_j <- object$X[[var]]
    denom_AB <- mean(fwl_AB$e_j[x_j == 1]) - mean(fwl_AB$e_j[x_j == 0])
    denom_BA <- mean(fwl_BA$e_j[x_j == 1]) - mean(fwl_BA$e_j[x_j == 0])

    est_AB <- if (abs(denom_AB) > 1e-10) raw_AB / denom_AB else raw_AB
    est_BA <- if (abs(denom_BA) > 1e-10) raw_BA / denom_BA else raw_BA

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

    Y_resid_AB <- .residualize_Y(object$X, object$Y, fs$idxA, hon_B, var)
    Y_resid_BA <- .residualize_Y(object$X, object$Y, fs$idxB, hon_A, var)

    slopes_AB <- .extract_curve_slopes(fs$rfA, object$X, Y_resid_AB,
                                       honest_idx = hon_B, var = var,
                                       grid = grid)
    slopes_BA <- .extract_curve_slopes(fs$rfB, object$X, Y_resid_BA,
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

  res <- honest_all(
    rf$forest, X_ord, y_hon, as.integer(honest_idx),
    bin_cols = as.integer(col_idx),
    cont_cols = as.integer(integer(0)),
    cont_thresh = numeric(0),
    per_leaf_denom = TRUE
  )

  res$binary$popavg[1]
}


#' @keywords internal
.extract_binary_multi_one_direction <- function(rf, X, Y, honest_idx, vars) {
  # Each variable needs its own residualization, so call single path
  out <- numeric(length(vars))
  for (j in seq_along(vars)) {
    out[j] <- .extract_binary_one_direction(rf, X, Y, honest_idx, vars[j])
  }
  names(out) <- vars
  out
}


#' @keywords internal
.honest_effect_binary_multi <- function(object, vars) {
  all_estimates <- matrix(0, nrow = object$honesty.splits, ncol = length(vars))
  colnames(all_estimates) <- vars

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]

    for (j in seq_along(vars)) {
      fwl_AB <- .residualize_FWL(object$X, object$Y, fs$idxA, fs$idxB, vars[j])
      fwl_BA <- .residualize_FWL(object$X, object$Y, fs$idxB, fs$idxA, vars[j])

      raw_AB <- .extract_binary_one_direction(fs$rfA, object$X, fwl_AB$Y_resid,
                                               honest_idx = fs$idxB, var = vars[j])
      raw_BA <- .extract_binary_one_direction(fs$rfB, object$X, fwl_BA$Y_resid,
                                               honest_idx = fs$idxA, var = vars[j])

      x_j <- object$X[[vars[j]]]
      denom_AB <- mean(fwl_AB$e_j[x_j == 1]) - mean(fwl_AB$e_j[x_j == 0])
      denom_BA <- mean(fwl_BA$e_j[x_j == 1]) - mean(fwl_BA$e_j[x_j == 0])

      est_AB <- if (abs(denom_AB) > 1e-10) raw_AB / denom_AB else raw_AB
      est_BA <- if (abs(denom_BA) > 1e-10) raw_BA / denom_BA else raw_BA

      all_estimates[r, j] <- (est_AB + est_BA) / 2
    }
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
