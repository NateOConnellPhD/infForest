#' Estimate Interactions Between Predictors
#'
#' Computes the effect of a focal predictor within subgroups defined by a
#' conditioning (\code{by}) variable. Returns subgroup-specific effects and
#' all pairwise differences. Supports binary x binary, continuous x binary,
#' and continuous x continuous interactions.
#'
#' @param object An \code{infForest} object fitted with \code{honesty = TRUE}.
#' @param var Character; name of the focal predictor variable.
#' @param by Character; name of the conditioning variable.
#' @param at Comparison points for the focal variable (continuous only).
#'   Default \code{c(0.25, 0.75)}. See \code{\link{effect}} for details.
#' @param type How to interpret \code{at}: \code{"quantile"} (default) or
#'   \code{"value"}.
#' @param by_at Subgroup definitions for a continuous \code{by} variable.
#'   A list of 2-element numeric vectors, each defining a quantile range
#'   (e.g., \code{list(c(0.10, 0.25), c(0.75, 0.90))}). Ignored for binary
#'   \code{by}. Default: \code{list(c(0.10, 0.25), c(0.75, 0.90))}.
#' @param bw Bandwidth for continuous focal variable estimation. Default 20.
#' @param q_lo,q_hi Grid bounds for the focal variable. Default 0.10 and 0.90.
#' @param subset Optional integer vector of observation indices to restrict
#'   estimation to. Intersected with the by-variable subgroups. Useful for
#'   higher-order interactions: \code{int(fit, "x2", by = "x6", subset = which(dat$x7 == 1))}.
#' @param ... Additional arguments (currently unused).
#'
#' @return A list of class \code{infForest_interaction} containing:
#' \describe{
#'   \item{variable}{Focal variable name.}
#'   \item{by}{Conditioning variable name.}
#'   \item{subgroups}{Data frame of subgroup-specific effects.}
#'   \item{differences}{Data frame of all pairwise differences.}
#' }
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#' int(fit, "x1", by = "x2")
#' int(fit, "x1", by = "x3", by_at = list(c(0.10, 0.33), c(0.33, 0.67), c(0.67, 0.90)))
#' int(fit, "x2", by = "x6", subset = which(dat$x7 == 1))
#' }
#'
#' @export
interaction <- function(object, ...) UseMethod("interaction")

#' @rdname interaction
#' @export
interaction.infForest <- function(object, var, by,
                                  at = c(0.25, 0.75),
                                  type = c("quantile", "value"),
                                  by_at = list(c(0.10, 0.25), c(0.75, 0.90)),
                                  bw = 20L,
                                  q_lo = 0.10, q_hi = 0.90,
                                  subset = NULL,
                                  variance = c("pasr", "sandwich", "both"),
                                  ci = TRUE, alpha = 0.05,
                                  p.value = FALSE,
                                  R_min = 20L, R_max = 200L,
                                  batch_size = 10L, tol = 0.05,
                                  n_stable = 2L, B_mc = 500L,
                                  nuisance = NULL, verbose = FALSE, ...) {

  check_infForest(object)
  check_varname(object, var)
  check_varname(object, by)

  type <- match.arg(type)
  variance <- match.arg(variance)

  by_var <- object$X[[by]]
  by_type <- detect_var_type(by_var)

  if (by_type == "binary") {
    result <- .interaction_by_binary(object, var, by, at = at, type = type,
                                     bw = bw, q_lo = q_lo, q_hi = q_hi,
                                     subset = subset,
                                     variance = variance, ci = ci, alpha = alpha,
                                     R_min = R_min, R_max = R_max,
                                     batch_size = batch_size, tol = tol,
                                     n_stable = n_stable, B_mc = B_mc,
                                     nuisance = nuisance, verbose = verbose)
  } else if (by_type == "continuous") {
    result <- .interaction_by_continuous(object, var, by, at = at, type = type,
                                         by_at = by_at, bw = bw,
                                         q_lo = q_lo, q_hi = q_hi,
                                         subset = subset)
  } else {
    stop("Categorical by-variables with >2 levels not yet supported.")
  }

  # P-value for interaction differences (H0: no effect modification)
  if (p.value && !is.null(result$differences)) {
    result$differences$p.value <- NA_real_
    for (k in seq_len(nrow(result$differences))) {
      diff_est <- result$differences$difference[k]
      diff_se <- if ("se" %in% names(result$differences)) result$differences$se[k] else NA_real_
      if (!is.na(diff_se) && diff_se > 0) {
        result$differences$p.value[k] <- 2 * pnorm(-abs(diff_est / diff_se))
      }
    }
  }
  result$show_p <- p.value

  result
}

#' @export
int <- function(...) interaction(...)


#' @keywords internal
.interaction_by_binary <- function(object, var, by, at, type, bw, q_lo, q_hi,
                                   subset = NULL,
                                   variance = "pasr", ci = TRUE, alpha = 0.05,
                                   R_min = 20L, R_max = 200L,
                                   batch_size = 10L, tol = 0.05,
                                   n_stable = 2L, B_mc = 500L,
                                   nuisance = NULL, verbose = FALSE) {

  by_var <- object$X[[by]]
  idx_1 <- which(by_var == 1)
  idx_0 <- which(by_var == 0)

  if (!is.null(subset)) {
    idx_1 <- intersect(idx_1, subset)
    idx_0 <- intersect(idx_0, subset)
  }

  focal_type <- detect_var_type(object$X[[var]])
  prop <- .fit_propensity(object$X, var, is_binary = (focal_type == "binary"))
  ghat <- prop$ghat

  eff_1 <- .effect_within_subset(object, var, subset_idx = idx_1,
                                  at = at, type = type, bw = bw,
                                  q_lo = q_lo, q_hi = q_hi, ghat = ghat)
  eff_0 <- .effect_within_subset(object, var, subset_idx = idx_0,
                                  at = at, type = type, bw = bw,
                                  q_lo = q_lo, q_hi = q_hi, ghat = ghat)

  subgroups <- data.frame(
    subgroup = c(paste0(by, " = 1"), paste0(by, " = 0")),
    estimate = c(eff_1, eff_0),
    stringsAsFactors = FALSE
  )

  differences <- data.frame(
    hi = paste0(by, " = 1"),
    lo = paste0(by, " = 0"),
    difference = eff_1 - eff_0,
    stringsAsFactors = FALSE
  )

  # Compute anchor description for printing
  if (focal_type == "continuous") {
    x_var <- object$X[[var]]
    if (type == "quantile") {
      at_vals <- quantile(x_var, probs = at, na.rm = TRUE)
      at_desc <- paste0("Q", round(at * 100))
    } else {
      at_vals <- at
      at_desc <- as.character(round(at, 2))
    }
    contrast_desc <- paste(at_desc[length(at_desc)], "-", at_desc[1])
  } else {
    at_vals <- NULL
    at_desc <- NULL
    contrast_desc <- NULL
  }

  out <- list(
    variable = var, by = by,
    var_type = focal_type, by_type = "binary",
    subgroups = subgroups, differences = differences,
    at = at, type = type, at_vals = at_vals,
    at_desc = at_desc, contrast_desc = contrast_desc
  )

  # --- CIs ---
  if (ci) {
    do_sandwich <- variance %in% c("sandwich", "both")
    do_pasr     <- variance %in% c("pasr", "both")
    z_crit <- qnorm(1 - alpha / 2)

    se_sand_diff <- NULL
    if (do_sandwich) {
      sand_1 <- .compute_sandwich_se(object, var, at = at, type = type,
                                      bw = bw, q_lo = q_lo, q_hi = q_hi,
                                      subset = idx_1, ghat = ghat)
      sand_0 <- .compute_sandwich_se(object, var, at = at, type = type,
                                      bw = bw, q_lo = q_lo, q_hi = q_hi,
                                      subset = idx_0, ghat = ghat)
      se_1 <- sand_1$se; se_0 <- sand_0$se
      se_sand_diff <- sqrt(se_1^2 + se_0^2)
      out$se_sandwich <- se_sand_diff
      out$subgroups$se <- c(se_1, se_0)
      out$subgroups$ci_lower <- out$subgroups$estimate - z_crit * c(se_1, se_0)
      out$subgroups$ci_upper <- out$subgroups$estimate + z_crit * c(se_1, se_0)
    }

    se_pasr_diff <- NULL
    if (do_pasr) {
      pasr_result <- .compute_pasr_int_se(
        object, var, by, at = at, type = type,
        bw = bw, q_lo = q_lo, q_hi = q_hi, subset = subset,
        R_min = R_min, R_max = R_max, batch_size = batch_size,
        tol = tol, n_stable = n_stable, B_mc = B_mc,
        nuisance = nuisance, verbose = verbose)
      se_pasr_diff <- pasr_result$se
      out$se_pasr <- se_pasr_diff
      out$C_psi <- pasr_result$C_psi
    }

    se_primary <- if (!is.null(se_pasr_diff)) se_pasr_diff else se_sand_diff
    out$se <- se_primary
    out$alpha <- alpha
    out$variance_method <- variance
    out$differences$se <- se_primary
    out$differences$ci_lower <- out$differences$difference - z_crit * se_primary
    out$differences$ci_upper <- out$differences$difference + z_crit * se_primary

    if (!is.null(se_pasr_diff) && !is.null(se_sand_diff))
      out$rho_V <- se_sand_diff^2 / se_pasr_diff^2
  }

  class(out) <- "infForest_interaction"
  out
}


#' @keywords internal
.interaction_by_continuous <- function(object, var, by, at, type, by_at,
                                       bw, q_lo, q_hi, subset = NULL) {

  by_var <- object$X[[by]]
  focal_type <- detect_var_type(object$X[[var]])

  # Fit propensity ONCE for the focal variable — shared across subgroups
  prop <- .fit_propensity(object$X, var, is_binary = (focal_type == "binary"))
  ghat <- prop$ghat

  n_groups <- length(by_at)
  group_labels <- character(n_groups)
  group_estimates <- numeric(n_groups)

  for (g in seq_len(n_groups)) {
    band <- by_at[[g]]
    if (length(band) != 2) stop("Each element of by_at must be a 2-element vector (lo, hi quantiles).")
    q_band_lo <- unname(quantile(by_var, band[1]))
    q_band_hi <- unname(quantile(by_var, band[2]))
    idx_g <- which(by_var >= q_band_lo & by_var <= q_band_hi)

    # Intersect with external subset if provided
    if (!is.null(subset)) {
      idx_g <- intersect(idx_g, subset)
    }

    if (length(idx_g) < 10) {
      warning(paste0("Subgroup ", by, " in [Q", round(band[1]*100), ", Q",
                     round(band[2]*100), "] has only ", length(idx_g),
                     " observations."))
    }

    group_labels[g] <- paste0(by, " in [Q", round(band[1]*100), ", Q", round(band[2]*100), "]")
    group_estimates[g] <- .effect_within_subset(object, var, subset_idx = idx_g,
                                                 at = at, type = type, bw = bw,
                                                 q_lo = q_lo, q_hi = q_hi,
                                                 ghat = ghat)
  }

  subgroups <- data.frame(
    subgroup = group_labels,
    estimate = group_estimates,
    stringsAsFactors = FALSE
  )

  # All pairwise differences
  pairs <- combn(n_groups, 2)
  n_pairs <- ncol(pairs)
  differences <- data.frame(
    hi = character(n_pairs),
    lo = character(n_pairs),
    difference = numeric(n_pairs),
    stringsAsFactors = FALSE
  )
  for (k in seq_len(n_pairs)) {
    i_hi <- pairs[2, k]
    i_lo <- pairs[1, k]
    differences$hi[k] <- group_labels[i_hi]
    differences$lo[k] <- group_labels[i_lo]
    differences$difference[k] <- group_estimates[i_hi] - group_estimates[i_lo]
  }

  # Compute anchor description for printing
  if (focal_type == "continuous") {
    x_var <- object$X[[var]]
    if (type == "quantile") {
      at_vals <- quantile(x_var, probs = at, na.rm = TRUE)
      at_desc <- paste0("Q", round(at * 100))
    } else {
      at_vals <- at
      at_desc <- as.character(round(at, 2))
    }
    contrast_desc <- paste(at_desc[length(at_desc)], "-", at_desc[1])
  } else {
    at_vals <- NULL
    at_desc <- NULL
    contrast_desc <- NULL
  }

  out <- list(
    variable = var,
    by = by,
    var_type = focal_type,
    by_type = "continuous",
    by_at = by_at,
    subgroups = subgroups,
    differences = differences,
    at = at, type = type, at_vals = at_vals,
    at_desc = at_desc, contrast_desc = contrast_desc
  )
  class(out) <- "infForest_interaction"
  out
}


#' @keywords internal
.effect_within_subset <- function(object, var, subset_idx, at, type,
                                   bw, q_lo, q_hi, ghat = NULL) {
  focal_type <- detect_var_type(object$X[[var]])

  if (focal_type == "binary") {
    return(.honest_effect_binary_subset(object, var, subset_idx, ghat = ghat))
  }

  x_var <- object$X[[var]]
  if (type == "quantile") {
    at_vals <- sort(unname(quantile(x_var, at)))
  } else {
    at_vals <- sort(at)
  }

  a <- at_vals[length(at_vals)]
  b <- at_vals[1]

  grid_lo <- min(a, b, unname(quantile(x_var, q_lo)))
  grid_hi <- max(a, b, unname(quantile(x_var, q_hi)))

  # Fit propensity once if not provided
  if (is.null(ghat)) {
    prop <- .fit_propensity(object$X, var, is_binary = FALSE)
    ghat <- prop$ghat
  }

  all_estimates <- numeric(object$honesty.splits)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    hon_AB <- intersect(fs$idxB, subset_idx)
    hon_BA <- intersect(fs$idxA, subset_idx)

    n_honest_sub <- length(hon_AB)

    # Guard: too few honest obs for a meaningful curve
    if (n_honest_sub < 4 || length(hon_BA) < 4) {
      all_estimates[r] <- NA_real_
      next
    }

    n_intervals <- max(1L, as.integer(n_honest_sub / bw))
    grid <- seq(grid_lo, grid_hi, length.out = n_intervals + 1)

    slopes_AB <- .extract_curve_slopes(fs$rfA, object$X, object$Y,
                                        honest_idx = hon_AB, var = var,
                                        grid = grid, ghat = ghat,
                                        object = object)
    slopes_BA <- .extract_curve_slopes(fs$rfB, object$X, object$Y,
                                        honest_idx = hon_BA, var = var,
                                        grid = grid, ghat = ghat,
                                        object = object)
    avg_slopes <- (slopes_AB + slopes_BA) / 2
    intervals <- diff(grid)
    curve_vals <- c(0, cumsum(avg_slopes * intervals))

    # Guard: need at least 2 non-NA curve values for interpolation
    if (sum(!is.na(curve_vals)) < 2) {
      all_estimates[r] <- NA_real_
      next
    }

    val_a <- approx(grid, curve_vals, xout = a, rule = 2)$y
    val_b <- approx(grid, curve_vals, xout = b, rule = 2)$y
    all_estimates[r] <- (val_a - val_b) / (a - b)
  }

  mean(all_estimates, na.rm = TRUE)
}


#' @keywords internal
.honest_effect_binary_subset <- function(object, var, subset_idx, ghat = NULL) {
  # Fit propensity once if not provided
  if (is.null(ghat)) {
    prop <- .fit_propensity(object$X, var, is_binary = TRUE)
    ghat <- prop$ghat
  }

  all_estimates <- numeric(object$honesty.splits)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    hon_AB <- intersect(fs$idxB, subset_idx)
    hon_BA <- intersect(fs$idxA, subset_idx)

    est_AB <- .extract_binary_one_direction(fs$rfA, object$X, object$Y,
                                             honest_idx = hon_AB, var = var,
                                             ghat = ghat, object = object)
    est_BA <- .extract_binary_one_direction(fs$rfB, object$X, object$Y,
                                             honest_idx = hon_BA, var = var,
                                             ghat = ghat, object = object)
    all_estimates[r] <- (est_AB + est_BA) / 2
  }

  mean(all_estimates)
}


#' Print method for infForest_interaction objects
#'
#' @param x An \code{infForest_interaction} object.
#' @param ... Additional arguments (ignored).
#' @export
print.infForest_interaction <- function(x, ...) {
  cat("Inference Forest Interaction\n")
  cat("  Variable:  ", x$variable, "\n")
  cat("  By:        ", x$by, "(", x$by_type, ")\n")
  if (x$var_type == "continuous" && !is.null(x$contrast_desc))
    cat("  Support:   ", x$contrast_desc, "\n")
  cat("\n")

  pct <- round((1 - (if (!is.null(x$alpha)) x$alpha else 0.05)) * 100)
  show_p <- isTRUE(x$show_p)

  .fmt_p <- function(p) {
    if (is.na(p)) return("NA")
    if (p < 0.001) return(sprintf("%.2e", p))
    sprintf("%.4f", p)
  }

  sub_w <- max(nchar(x$subgroups$subgroup))
  diff_hi_w <- max(nchar(x$differences$hi))
  diff_lo_w <- max(nchar(x$differences$lo))

  unit_label <- if (x$var_type == "continuous") "  (per unit)" else ""

  cat("  Subgroup effects:\n")
  for (k in seq_len(nrow(x$subgroups))) {
    line <- sprintf("    %-*s  %8.4f%s",
                    sub_w, x$subgroups$subgroup[k],
                    x$subgroups$estimate[k], unit_label)
    if ("se" %in% names(x$subgroups) && !is.na(x$subgroups$se[k]))
      line <- paste0(line, sprintf("  (SE: %.4f, %d%% CI: [%.4f, %.4f])",
                                    x$subgroups$se[k], pct,
                                    x$subgroups$ci_lower[k], x$subgroups$ci_upper[k]))
    cat(line, "\n")
  }

  cat("\n  Pairwise differences:\n")
  for (k in seq_len(nrow(x$differences))) {
    line <- sprintf("    %-*s  vs  %-*s  %8.4f",
                    diff_hi_w, x$differences$hi[k],
                    diff_lo_w, x$differences$lo[k],
                    x$differences$difference[k])
    if ("se" %in% names(x$differences) && !is.na(x$differences$se[k]))
      line <- paste0(line, sprintf("  (SE: %.4f, %d%% CI: [%.4f, %.4f])",
                                    x$differences$se[k], pct,
                                    x$differences$ci_lower[k], x$differences$ci_upper[k]))
    if (show_p && "p.value" %in% names(x$differences) && !is.na(x$differences$p.value[k]))
      line <- paste0(line, sprintf("  p: %s", .fmt_p(x$differences$p.value[k])))
    cat(line, "\n")
  }

  if (!is.null(x$rho_V))
    cat(sprintf("\n  rho_V: %.2f\n", x$rho_V))

  invisible(x)
}

