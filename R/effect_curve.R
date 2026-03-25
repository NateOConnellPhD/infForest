#' Nonlinear Effect Curve for a Continuous Predictor
#'
#' Estimates how a continuous predictor relates to the outcome across its
#' support. Two visualization modes:
#'
#' \strong{Slope curve} (default): Local AIPW slopes at each grid interval
#' with pointwise sandwich bands. Shows where the effect is strongest,
#' weakest, or changes direction. Flat = linear; changing = nonlinear;
#' crossing zero = no effect in that region. Fast (no extra computation).
#'
#' \strong{Level curve} (\code{type = "level"}): Adjusted mean E[Y | X_j = x]
#' at each grid point with PASR bands. The forest analog of a GLM fitted
#' curve with CI bands. Requires PASR resampling (slower).
#'
#' @param object An \code{infForest} object.
#' @param var Character; name of a continuous predictor variable.
#' @param q_lo,q_hi Quantiles defining the grid bounds. Default 0.10 and 0.90.
#' @param bw Bandwidth: target honest observations per grid interval. Default 20.
#' @param type \code{"slope"} (default) or \code{"level"}.
#' @param alpha Significance level. Default 0.05.
#' @param propensity_trees Trees for the propensity model. Default 2000.
#' @param R Number of PASR replicates for type = "level". Default 50.
#' @param ... Additional arguments passed to PASR internals.
#'
#' @return A list of class \code{infForest_curve}.
#'
#' @export
effect_curve <- function(object, ...) UseMethod("effect_curve")

#' @rdname effect_curve
#' @export
effect_curve.infForest <- function(object, var, q_lo = 0.10, q_hi = 0.90,
                                   bw = 20L, type = c("level","slope"),
                                   alpha = 0.05, propensity_trees = 2000L,
                                   R = 50L, ...) {
  check_infForest(object)
  check_varname(object, var)
  type <- match.arg(type)
  pasr <- object$pasr

  x_var <- object$X[[var]]
  if (detect_var_type(x_var) != "continuous")
    stop("effect_curve() requires a continuous predictor.")

  n <- nrow(object$X)
  Y <- object$Y
  z_crit <- qnorm(1 - alpha / 2)

  lo <- unname(quantile(x_var, q_lo))
  hi <- unname(quantile(x_var, q_hi))
  n_hon <- n %/% 2
  G <- max(5L, min(20L, as.integer(n_hon / bw)))
  grid <- seq(lo, hi, length.out = G + 1)

  # Propensity once
  ghat <- .fit_propensity(object$X, var, is_binary = FALSE,
                           n_trees = propensity_trees)$ghat

  if (type == "slope") {
    out <- .effect_curve_slope(object, var, grid, G, ghat, n, Y, alpha, z_crit)
  } else {
    if (!is.null(pasr)) {
      out <- .effect_curve_level_pasr(object, var, grid, G, ghat, x_var,
                                       alpha, z_crit, pasr)
    } else {
      out <- .effect_curve_level(object, var, grid, G, ghat, n, Y, x_var,
                                  alpha, z_crit, propensity_trees, R)
    }
  }

  out$variable <- var
  out$grid <- grid
  out$n_intervals <- G
  out$q_lo <- q_lo
  out$q_hi <- q_hi
  out$alpha <- alpha
  out$type <- type

  out$df <- data.frame(
    variable = var,
    type = "continuous",
    estimand = if (type == "slope") "slope" else "mean",
    level = as.character(round(
      if (type == "slope") (grid[-1] + grid[-(G+1)]) / 2 else grid, 4)),
    estimate = out$estimate,
    se = out$se,
    ci_lower = out$ci_lower,
    ci_upper = out$ci_upper,
    p.value = NA_real_,
    stringsAsFactors = FALSE
  )

  class(out) <- "infForest_curve"
  out
}


# ============================================================
# Slope curve: sandwich SE from phi_scores
# ============================================================
#' @keywords internal
.effect_curve_slope <- function(object, var, grid, G, ghat, n, Y, alpha, z_crit) {

  # Accumulate phi_scores across honesty splits
  phi_sum <- matrix(0, nrow = n, ncol = G)
  phi_cnt <- matrix(0L, nrow = n, ncol = G)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]

    for (dir in list(
      list(rf = fs$rfA, hon = fs$idxB),
      list(rf = fs$rfB, hon = fs$idxA)
    )) {
      X_ord <- .get_X_ord(object, dir$rf)
      col_idx <- get_ranger_col_idx(dir$rf, var)
      y_hon <- rep(NA_real_, n)
      y_hon[dir$hon] <- as.numeric(Y[dir$hon])

      res <- aipw_curve_v2_cpp(dir$rf$forest, X_ord, y_hon,
                                as.integer(dir$hon), ghat, col_idx, grid)

      ps <- res$phi_scores  # n_hon x G
      hon <- dir$hon

      for (j in seq_along(hon)) {
        k <- hon[j]
        for (g in seq_len(G)) {
          v <- ps[j, g]
          if (!is.na(v)) {
            phi_sum[k, g] <- phi_sum[k, g] + v
            phi_cnt[k, g] <- phi_cnt[k, g] + 1L
          }
        }
      }
    }
  }

  # Sandwich SE at each interval
  slopes <- numeric(G)
  se <- numeric(G)

  for (g in seq_len(G)) {
    valid <- phi_cnt[, g] > 0
    nv <- sum(valid)
    if (nv > 1) {
      phi_avg <- phi_sum[valid, g] / phi_cnt[valid, g]
      psi_bar <- mean(phi_avg)
      V_IF <- sum((phi_avg - psi_bar)^2) / (nv * (nv - 1))
      slopes[g] <- psi_bar
      se[g] <- sqrt(V_IF)
    }
  }

  ci_lower <- slopes - z_crit * se
  ci_upper <- slopes + z_crit * se

  list(estimate = slopes, se = se, ci_lower = ci_lower, ci_upper = ci_upper)
}


# ============================================================
# Level curve: PASR SE from repeated evaluations
# ============================================================
#' @keywords internal
.effect_curve_level <- function(object, var, grid, G, ghat, n, Y, x_var,
                                 alpha, z_crit, propensity_trees, R) {
  G1 <- G + 1

  # Step 1: compute the observed adjusted means at each grid point
  mu_obs <- .compute_level_curve_once(object, var, grid, G, ghat, n, Y, x_var)

  # Step 2: PASR — fit nuisance, generate synthetic Y, repeat
  fhat_full <- .get_full_forest_predictions(object)
  resid <- as.numeric(Y) - fhat_full
  sigma2_hat <- mean(resid^2, na.rm = TRUE)

  mu_matrix <- matrix(NA_real_, nrow = R, ncol = G1)

  for (r in seq_len(R)) {
    Y_syn <- fhat_full + rnorm(n, 0, sqrt(sigma2_hat))
    mu_r <- .compute_level_curve_once(object, var, grid, G, ghat, n,
                                       Y_syn, x_var)
    mu_matrix[r, ] <- mu_r
  }

  # PASR SE: SD of replicate-level estimates
  se <- apply(mu_matrix, 2, sd)

  ci_lower <- mu_obs - z_crit * se
  ci_upper <- mu_obs + z_crit * se

  list(estimate = mu_obs, se = se, ci_lower = ci_lower, ci_upper = ci_upper)
}


# ============================================================
# Compute adjusted mean curve once (used by observed and PASR paths)
# ============================================================
#' @keywords internal
.compute_level_curve_once <- function(object, var, grid, G, ghat, n, Y, x_var) {
  G1 <- G + 1
  col_idx <- get_ranger_col_idx(object$forests[[1]]$rfA, var)
  caches <- object$forest_caches
  use_cached <- !is.null(caches)

  if (use_cached) {
    # Fast path: use precomputed forest caches
    cache_names <- names(caches)
    n_dirs <- length(cache_names)
    mu_splits <- matrix(NA_real_, nrow = n_dirs, ncol = G1)

    for (d in seq_along(cache_names)) {
      cache <- caches[[cache_names[d]]]
      hon <- as.integer(cache$honest_idx)
      n_hon <- length(hon)

      res <- aipw_curve_cached_cpp(cache, ghat, col_idx, grid)
      fg <- res$fhat_grid
      fo <- cache$fhat_obs
      sigma2 <- res$sigma2_ej

      for (g in seq_len(G1)) {
        phi_sum <- 0; cnt <- 0L
        for (j in seq_along(hon)) {
          k <- hon[j]
          fg_jg <- fg[j, g]; fo_j <- fo[j]
          if (is.na(fg_jg) || is.na(fo_j)) next
          omega_k <- (x_var[k] - ghat[k]) / sigma2
          R_k <- as.numeric(Y[k]) - fo_j
          phi_sum <- phi_sum + fg_jg + omega_k * R_k
          cnt <- cnt + 1L
        }
        mu_splits[d, g] <- if (cnt > 0) phi_sum / cnt else NA_real_
      }
    }
    return(colMeans(mu_splits, na.rm = TRUE))
  }

  # Fallback: non-cached path
  n_splits <- length(object$forests)
  mu_splits <- matrix(NA_real_, nrow = n_splits * 2, ncol = G1)

  d <- 0L
  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]

    for (dir in list(
      list(rf = fs$rfA, hon = fs$idxB),
      list(rf = fs$rfB, hon = fs$idxA)
    )) {
      d <- d + 1L
      X_ord <- .get_X_ord(object, dir$rf)
      col_idx_d <- get_ranger_col_idx(dir$rf, var)
      y_hon <- rep(NA_real_, n)
      y_hon[dir$hon] <- as.numeric(Y[dir$hon])
      hon <- dir$hon

      res <- aipw_curve_v2_cpp(dir$rf$forest, X_ord, y_hon,
                                as.integer(hon), ghat, col_idx_d, grid)

      fg <- res$fhat_grid
      fo <- res$fhat_obs
      sigma2 <- res$sigma2_ej

      for (g in seq_len(G1)) {
        phi_sum <- 0; cnt <- 0L
        for (j in seq_along(hon)) {
          k <- hon[j]
          fg_jg <- fg[j, g]
          fo_j <- fo[j]
          if (is.na(fg_jg) || is.na(fo_j)) next
          omega_k <- (x_var[k] - ghat[k]) / sigma2
          R_k <- as.numeric(Y[k]) - fo_j
          phi_sum <- phi_sum + fg_jg + omega_k * R_k
          cnt <- cnt + 1L
        }
        mu_splits[d, g] <- if (cnt > 0) phi_sum / cnt else NA_real_
      }
    }
  }

  colMeans(mu_splits, na.rm = TRUE)
}


# ============================================================
# Level curve using pre-fit PASR object
# ============================================================
#' @keywords internal
.effect_curve_level_pasr <- function(object, var, grid, G, ghat, x_var,
                                      alpha, z_crit, pasr_obj) {
  G1 <- G + 1

  mu_obs <- .compute_level_curve_once(object, var, grid, G, ghat,
                                       nrow(object$X), object$Y, x_var)

  col_idx <- get_ranger_col_idx(object$forests[[1]]$rfA, var)
  nt <- .get_n_threads()

  # Build Y_syn list
  Y_syn_list <- vector("list", pasr_obj$R)
  for (r in seq_len(pasr_obj$R)) Y_syn_list[[r]] <- pasr_obj$Y_syn[, r]

  # Single C++ batch call
  mu_matrix <- pasr_extract_all_level_curve_cpp(
    pasr_obj$caches, Y_syn_list, ghat, x_var,
    col_idx, grid, n_threads = nt)

  se <- apply(mu_matrix, 2, sd)
  ci_lower <- mu_obs - z_crit * se
  ci_upper <- mu_obs + z_crit * se

  list(estimate = mu_obs, se = se, ci_lower = ci_lower, ci_upper = ci_upper)
}


# ============================================================
# Helper: full forest predictions for PASR nuisance
# ============================================================
#' @keywords internal
.get_full_forest_predictions <- function(object) {
  n <- nrow(object$X)
  pred_sum <- numeric(n)
  pred_cnt <- integer(n)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    for (dir in list(
      list(rf = fs$rfA, hon = fs$idxB),
      list(rf = fs$rfB, hon = fs$idxA)
    )) {
      X_ord <- .get_X_ord(object, dir$rf)
      y_hon <- rep(NA_real_, n)
      y_hon[dir$hon] <- as.numeric(object$Y[dir$hon])

      preds <- honest_predict_cpp(dir$rf$forest, X_ord, X_ord,
                                   y_hon, as.integer(dir$hon))
      for (k in dir$hon) {
        if (!is.na(preds[k])) {
          pred_sum[k] <- pred_sum[k] + preds[k]
          pred_cnt[k] <- pred_cnt[k] + 1L
        }
      }
    }
  }

  valid <- pred_cnt > 0
  out <- rep(NA_real_, n)
  out[valid] <- pred_sum[valid] / pred_cnt[valid]
  out
}


#' @export
print.infForest_curve <- function(x, ...) {
  cat("Inference Forest Effect Curve\n")
  cat("  Variable:   ", x$variable, "\n")
  cat("  Type:       ", x$type, "\n")
  cat("  Grid range: ", round(min(x$grid), 3), "to", round(max(x$grid), 3), "\n")
  cat("  Intervals:  ", x$n_intervals, "\n")
  if (x$type == "slope") {
    cat("  Slope range:", round(min(x$estimate), 4), "to",
        round(max(x$estimate), 4), "\n")
  } else {
    cat("  Mean range: ", round(min(x$estimate), 4), "to",
        round(max(x$estimate), 4), "\n")
  }
  cat("  Max SE:     ", round(max(x$se), 4), "\n")
  cat("  Variance:   ", if (x$type == "slope") "sandwich" else "PASR", "\n")
  cat("  CI level:   ", round((1 - x$alpha) * 100), "%\n")
  invisible(x)
}


#' @export
plot.infForest_curve <- function(x, ...) {
  yl <- range(c(x$ci_lower, x$ci_upper), na.rm = TRUE)
  yl <- yl + c(-0.05, 0.05) * diff(yl)

  if (x$type == "slope") {
    midpoints <- (x$grid[-1] + x$grid[-length(x$grid)]) / 2
    xvals <- midpoints
    ylab <- "Local slope (per unit)"
  } else {
    xvals <- x$grid
    ylab <- "Adjusted mean"
  }

  plot(xvals, x$estimate, type = "l", lwd = 2,
       xlab = x$variable, ylab = ylab,
       main = paste("Effect curve:", x$variable),
       ylim = yl, ...)
  polygon(c(xvals, rev(xvals)),
          c(x$ci_lower, rev(x$ci_upper)),
          col = rgb(0, 0, 0, 0.12), border = NA)
  lines(xvals, x$estimate, lwd = 2)
  if (x$type == "slope") abline(h = 0, lty = 2, col = "gray50")
}
