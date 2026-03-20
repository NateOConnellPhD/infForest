#' Estimate the Effect of a Predictor
#'
#' Computes population-averaged effects of a predictor using honest AIPW
#' estimation. For binary predictors, uses leaf augmentation to ensure every
#' tree contributes a nonzero prediction contrast, then applies the standard
#' AIPW scorer. For continuous predictors, returns per-unit slopes from the
#' AIPW counterfactual curve without augmentation.
#'
#' The estimator combines honest forest predictions (working model) with
#' propensity-weighted honest residuals (debiasing correction). Double
#' robustness ensures consistency if either the forest or propensity model
#' is consistent. The estimator achieves the semiparametric efficiency bound.
#'
#' @param object An \code{infForest} object.
#' @param var Character; name of the predictor variable.
#' @param at Numeric vector of comparison points for continuous predictors.
#'   Default \code{c(0.25, 0.75)}. Interpretation depends on \code{type}.
#'   All pairwise contrasts are returned. Ignored for binary predictors.
#' @param type How to interpret \code{at}: \code{"quantile"} (default) treats
#'   values as quantile probabilities, \code{"value"} treats them as raw values.
#' @param q_lo,q_hi Quantiles defining the grid bounds for curve-based
#'   estimation (continuous predictors). Default 0.10 and 0.90.
#' @param bw Bandwidth: target number of honest observations per grid interval.
#'   Controls grid density. Default 20.
#' @param subset Optional integer vector of observation indices to restrict
#'   honest estimation to. The forest routing is unchanged; only the specified
#'   observations contribute AIPW scores. Useful for conditional resolution.
#' @param propensity_trees Number of trees for the propensity model. Default 2000.
#' @param ... Additional arguments (currently unused).
#'
#' @return A list of class \code{infForest_effect}.
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#' effect(fit, "treatment")
#' effect(fit, "age")
#' effect(fit, "age", at = c(0.10, 0.50, 0.90))
#' }
#'
#' @export
effect <- function(object, ...) UseMethod("effect")

#' @rdname effect
#' @export
effect.infForest <- function(object, var, at = c(0.25, 0.75),
                             type = c("quantile", "value"),
                             q_lo = 0.10, q_hi = 0.90,
                             bw = 20L, subset = NULL,
                             propensity_trees = 2000L,
                             ghat = NULL, ...) {

  check_infForest(object)
  check_varname(object, var)

  type <- match.arg(type)
  x_var <- object$X[[var]]
  var_type <- detect_var_type(x_var)

  if (var_type == "binary") {
    est <- .aipw_effect_binary(object, var, subset = subset,
                                propensity_trees = propensity_trees,
                                ghat = ghat)
    out <- list(
      variable = var,
      var_type = var_type,
      estimate = est$psi,
      diagnostics = est$diagnostics,
      n_intervals = 1L,
      subset = subset
    )
    class(out) <- "infForest_effect"
    return(out)
  }

  if (var_type != "continuous") {
    stop("Categorical predictors with >2 levels not yet supported.")
  }

  if (type == "quantile") {
    at_vals <- unname(quantile(x_var, at))
    at_labels <- paste0("Q", round(at * 100))
  } else {
    at_vals <- at
    at_labels <- as.character(round(at, 3))
  }

  at_vals <- sort(at_vals)
  at_labels <- at_labels[order(at)]

  n_honest <- nrow(object$X) %/% 2
  n_intervals <- max(1L, as.integer(n_honest / bw))

  grid_lo <- min(at_vals, unname(quantile(x_var, q_lo)))
  grid_hi <- max(at_vals, unname(quantile(x_var, q_hi)))

  curve_result <- .aipw_build_curve(object, var, grid_lo, grid_hi,
                                     n_honest = n_honest, bw = bw,
                                     subset = subset,
                                     propensity_trees = propensity_trees,
                                     ghat = ghat)

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
    contrasts_df$estimate[k] <- (val_hi - val_lo) / (at_vals[i_hi] - at_vals[i_lo])
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


# ============================================================
# AIPW internals
# ============================================================

#' Get ranger-ordered X matrix, using cache on infForest object if available
#' @keywords internal
.get_X_ord <- function(object, rf) {
  # Check if cached on the object (set by infForest() at fit time)
  if (!is.null(object$X_ord)) return(object$X_ord)
  # Fallback: compute from scratch
  reorder_X_to_ranger(object$X, rf)
}

#' Fit propensity model: predict X_j from X_{-j} using penalized regression
#'
#' For continuous X_j: ridge regression (alpha=0), penalty by CV.
#' For binary X_j: logistic ridge (alpha=0, family="binomial"), penalty by CV.
#'
#' Ridge avoids the spurious variation of forest OOB propensities at small n.
#' When X_j is independent of X_{-j}, the penalty shrinks all coefficients
#' toward zero, returning ghat ≈ constant (the marginal mean/prevalence).
#' When X_j is confounded, the real association survives the penalty.
#'
#' @keywords internal
.fit_propensity <- function(X, var, is_binary, n_trees = NULL) {
  var_col_r <- which(names(X) == var)
  X_minus_j <- as.matrix(X[, -var_col_r, drop = FALSE])
  x_j <- X[[var]]

  if (is_binary) {
    cv_fit <- glmnet::cv.glmnet(X_minus_j, x_j, alpha = 0,
                                 family = "binomial", nfolds = 5)
    ghat <- as.numeric(predict(cv_fit, X_minus_j,
                                s = "lambda.min", type = "response"))
    ghat <- pmax(pmin(ghat, 0.975), 0.025)
  } else {
    cv_fit <- glmnet::cv.glmnet(X_minus_j, x_j, alpha = 0, nfolds = 5)
    ghat <- as.numeric(predict(cv_fit, X_minus_j, s = "lambda.min"))
  }

  list(fit = cv_fit, ghat = ghat)
}


#' AIPW scores for one fold direction — v2: no counterfactual matrices,
#' fused augmentation for binary.
#' @keywords internal
.aipw_one_direction <- function(rf, X_ord, Y, honest_idx, var, col_idx, a, b,
                                 is_binary, ghat, subset = NULL) {
  n <- nrow(X_ord)

  y_hon <- rep(NA_real_, n)
  hon_use <- if (!is.null(subset)) intersect(honest_idx, subset) else honest_idx
  y_hon[hon_use] <- as.numeric(Y[hon_use])

  res <- aipw_scores_v2_cpp(
    forest = rf$forest,
    X_obs = X_ord,
    y_honest = y_hon,
    honest_idx = as.integer(hon_use),
    ghat = ghat,
    var_col = col_idx,
    is_binary = is_binary,
    a = a,
    b = b
  )

  res
}


#' Full AIPW binary effect with cross-fitting and repeated honest splits
#' @keywords internal
.aipw_effect_binary <- function(object, var, subset = NULL,
                                 propensity_trees = 2000L,
                                 ghat = NULL) {
  psi_splits <- numeric(object$honesty.splits)
  diag_list <- vector("list", object$honesty.splits)

  if (is.null(ghat)) {
    prop <- .fit_propensity(object$X, var, is_binary = TRUE,
                             n_trees = propensity_trees)
    ghat <- prop$ghat
  }

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]

    # Compute X_ord and col_idx once per forest pair
    X_ord_A <- .get_X_ord(object, fs$rfA)
    col_idx_A <- get_ranger_col_idx(fs$rfA, var)
    X_ord_B <- .get_X_ord(object, fs$rfB)
    col_idx_B <- get_ranger_col_idx(fs$rfB, var)

    res_AB <- .aipw_one_direction(
      rf = fs$rfA, X_ord = X_ord_A, Y = object$Y,
      honest_idx = fs$idxB, var = var, col_idx = col_idx_A,
      a = 1, b = 0, is_binary = TRUE,
      ghat = ghat, subset = subset
    )

    res_BA <- .aipw_one_direction(
      rf = fs$rfB, X_ord = X_ord_B, Y = object$Y,
      honest_idx = fs$idxA, var = var, col_idx = col_idx_B,
      a = 1, b = 0, is_binary = TRUE,
      ghat = ghat, subset = subset
    )

    psi_splits[r] <- (res_AB$psi + res_BA$psi) / 2
    diag_list[[r]] <- list(AB = res_AB, BA = res_BA)
  }

  list(
    psi = mean(psi_splits),
    per_split = psi_splits,
    diagnostics = diag_list
  )
}


#' Build AIPW effect curve for continuous predictors (no augmentation)
#' @keywords internal
.aipw_build_curve <- function(object, var, grid_lo, grid_hi, n_honest, bw,
                               subset = NULL, propensity_trees = 2000L,
                               ghat = NULL) {
  n_intervals <- max(1L, as.integer(n_honest / bw))
  grid <- seq(grid_lo, grid_hi, length.out = n_intervals + 1)

  if (is.null(ghat)) {
    prop <- .fit_propensity(object$X, var, is_binary = FALSE,
                             n_trees = propensity_trees)
    ghat <- prop$ghat
  }

  all_slopes <- matrix(0, nrow = object$honesty.splits, ncol = n_intervals)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]

    X_ord_A <- .get_X_ord(object, fs$rfA)
    col_idx_A <- get_ranger_col_idx(fs$rfA, var)
    X_ord_B <- .get_X_ord(object, fs$rfB)
    col_idx_B <- get_ranger_col_idx(fs$rfB, var)

    slopes_AB <- .aipw_curve_one_direction(
      rf = fs$rfA, X_ord = X_ord_A, Y = object$Y,
      honest_idx = fs$idxB, col_idx = col_idx_A,
      grid = grid, ghat = ghat, subset = subset
    )

    slopes_BA <- .aipw_curve_one_direction(
      rf = fs$rfB, X_ord = X_ord_B, Y = object$Y,
      honest_idx = fs$idxA, col_idx = col_idx_B,
      grid = grid, ghat = ghat, subset = subset
    )

    all_slopes[r, ] <- (slopes_AB + slopes_BA) / 2
  }

  avg_slopes <- colMeans(all_slopes)
  intervals <- diff(grid)
  curve_vals <- c(0, cumsum(avg_slopes * intervals))

  list(grid = grid, curve = curve_vals, slopes = avg_slopes, intervals = intervals)
}


#' AIPW curve slopes via aipw_curve_v2_cpp — grid vector, no matrix copies
#' @keywords internal
.aipw_curve_one_direction <- function(rf, X_ord, Y, honest_idx, col_idx, grid,
                                       ghat, subset = NULL) {
  n <- nrow(X_ord)
  G <- length(grid) - 1

  y_hon <- rep(NA_real_, n)
  hon_use <- if (!is.null(subset)) intersect(honest_idx, subset) else honest_idx
  y_hon[hon_use] <- as.numeric(Y[hon_use])

  # v2: pass grid vector directly — no matrix list allocation
  res <- aipw_curve_v2_cpp(
    forest = rf$forest,
    X_obs = X_ord,
    y_honest = y_hon,
    honest_idx = as.integer(hon_use),
    ghat = ghat,
    var_col = col_idx,
    grid_points = grid
  )

  slopes <- res$slopes
  slopes[is.na(slopes)] <- 0
  slopes
}


# ============================================================
# Legacy wrappers (interaction.R calls these)
# ============================================================

#' @keywords internal
.honest_effect_binary <- function(object, var, subset = NULL) {
  .aipw_effect_binary(object, var, subset = subset)$psi
}

#' @keywords internal
.honest_build_curve <- function(object, var, grid_lo, grid_hi, n_honest, bw,
                                subset = NULL) {
  .aipw_build_curve(object, var, grid_lo, grid_hi, n_honest, bw,
                     subset = subset)
}

#' @keywords internal
.honest_effect_binary_multi <- function(object, vars) {
  out <- numeric(length(vars))
  for (j in seq_along(vars)) {
    out[j] <- .aipw_effect_binary(object, vars[j])$psi
  }
  names(out) <- vars
  out
}

#' @keywords internal
.extract_binary_one_direction <- function(rf, X, Y, honest_idx, var,
                                           ghat = NULL, object = NULL) {
  X_df <- if (is.data.frame(X)) X else as.data.frame(X)
  if (is.null(ghat)) {
    prop <- .fit_propensity(X_df, var, is_binary = TRUE)
    ghat <- prop$ghat
  }
  if (!is.null(object)) {
    X_ord <- .get_X_ord(object, rf)
  } else {
    X_ord <- reorder_X_to_ranger(X_df, rf)
  }
  col_idx <- get_ranger_col_idx(rf, var)
  res <- .aipw_one_direction(
    rf = rf, X_ord = X_ord, Y = Y,
    honest_idx = honest_idx, var = var, col_idx = col_idx,
    a = 1, b = 0, is_binary = TRUE,
    ghat = ghat
  )
  res$psi
}

#' @keywords internal
.extract_curve_slopes <- function(rf, X, Y, honest_idx, var, grid,
                                   ghat = NULL, object = NULL) {
  X_df <- if (is.data.frame(X)) X else as.data.frame(X)
  if (is.null(ghat)) {
    prop <- .fit_propensity(X_df, var, is_binary = FALSE)
    ghat <- prop$ghat
  }
  if (!is.null(object)) {
    X_ord <- .get_X_ord(object, rf)
  } else {
    X_ord <- reorder_X_to_ranger(X_df, rf)
  }
  col_idx <- get_ranger_col_idx(rf, var)
  .aipw_curve_one_direction(
    rf = rf, X_ord = X_ord, Y = Y,
    honest_idx = honest_idx, col_idx = col_idx,
    grid = grid, ghat = ghat
  )
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

  est <- if (x$var_type == "binary") x$estimate else x$contrasts$estimate[1]
  z <- qnorm(1 - (if (!is.null(x$alpha)) x$alpha else 0.05) / 2)

  if (x$var_type == "binary") {
    cat("  Estimate:   ", round(x$estimate, 4), "\n")
    if (!is.null(x$se)) {
      se_label <- if (!is.null(x$variance_method)) paste0("SE (", x$variance_method, ")") else "SE"
      cat(sprintf("  %-12s %.4f\n", paste0(se_label, ":"), x$se))
      cat("  95% CI:      [", round(x$ci_lower, 4), ", ", round(x$ci_upper, 4), "]\n")
      cat("  p-value:    ", format.pval(x$pval, digits = 3), "\n")
      if (!is.null(x$se_sandwich) && !is.null(x$se_pasr)) {
        cat(sprintf("  SE (sand):   %.4f  |  95%% CI: [%.4f, %.4f]\n",
                    x$se_sandwich, est - z * x$se_sandwich, est + z * x$se_sandwich))
        cat(sprintf("  rho_V:       %.2f\n", x$rho_V))
      }
    }
  } else {
    cat("  Intervals:  ", x$n_intervals, "\n")
    cat("\n  Pairwise contrasts (per unit):\n")
    df <- x$contrasts
    for (k in seq_len(nrow(df))) {
      cat(sprintf("    %s to %s  [%.3f, %.3f]:  %.4f\n",
                  df$lo[k], df$hi[k], df$lo_val[k], df$hi_val[k], df$estimate[k]))
    }
    if (!is.null(x$se)) {
      se_label <- if (!is.null(x$variance_method)) paste0("SE (", x$variance_method, ")") else "SE"
      cat(sprintf("\n  %-12s %.4f\n", paste0(se_label, ":"), x$se))
      cat(sprintf("  95%% CI: [%.4f, %.4f]\n", x$ci_lower, x$ci_upper))
      cat(sprintf("  p-value: %s\n", format.pval(x$pval, digits = 3)))
      if (!is.null(x$se_sandwich) && !is.null(x$se_pasr)) {
        cat(sprintf("  SE (sand): %.4f  |  95%% CI: [%.4f, %.4f]\n",
                    x$se_sandwich, est - z * x$se_sandwich, est + z * x$se_sandwich))
        cat(sprintf("  rho_V: %.2f\n", x$rho_V))
      }
    }
  }
  invisible(x)
}
