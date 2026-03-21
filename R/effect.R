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
                             variance = c("sandwich", "pasr", "both"),
                             ci = TRUE, alpha = 0.05,
                             propensity_trees = 2000L,
                             ghat = NULL,
                             R_min = 20L, R_max = 200L,
                             batch_size = 10L, tol = 0.05,
                             n_stable = 2L, B_mc = 500L,
                             nuisance = NULL, verbose = FALSE, ...) {

  check_infForest(object)
  check_varname(object, var)

  type <- match.arg(type)
  variance <- match.arg(variance)
  x_var <- object$X[[var]]
  var_type <- detect_var_type(x_var)
  is_bin <- (var_type == "binary")

  # Fit propensity once, shared by point estimate and variance estimation
  # (categorical handles propensity per-contrast internally)
  if (is.null(ghat) && var_type != "categorical") {
    prop <- .fit_propensity(object$X, var, is_binary = is_bin,
                            n_trees = propensity_trees)
    ghat <- prop$ghat
  }

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
    est_scalar <- est$psi

  } else if (var_type == "continuous") {

    if (type == "quantile") {
      at_vals <- unname(quantile(x_var, at))
      at_labels <- paste0("Q", round(at * 100))
    } else {
      at_vals <- at
      at_labels <- as.character(round(at, 3))
    }

    # For the grid, we need sorted values; but preserve user's contrast direction
    at_vals_sorted <- sort(at_vals)
    at_order <- order(at)

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

    if (n_at == 2) {
      # Two points: user specifies direction. at[1] is "from", at[2] is "to".
      # Contrast = f(to) - f(from), slope = contrast / (to - from)
      val_from <- approx(curve_result$grid, curve_result$curve, xout = at_vals[1], rule = 2)$y
      val_to   <- approx(curve_result$grid, curve_result$curve, xout = at_vals[2], rule = 2)$y
      contrasts_df <- data.frame(
        contrast = paste0(at_labels[2], " - ", at_labels[1]),
        from_val = at_vals[1],
        to_val = at_vals[2],
        estimate = (val_to - val_from) / (at_vals[2] - at_vals[1]),
        stringsAsFactors = FALSE
      )
    } else {
      # 3+ points: all pairwise, ascending direction
      pairs <- combn(n_at, 2)
      n_pairs <- ncol(pairs)
      # Sort at_vals for consistent pairwise ordering
      sv <- sort(at_vals)
      sl <- at_labels[order(at_vals)]
      contrasts_df <- data.frame(
        contrast = character(n_pairs),
        from_val = numeric(n_pairs),
        to_val = numeric(n_pairs),
        estimate = numeric(n_pairs),
        stringsAsFactors = FALSE
      )
      for (k in seq_len(n_pairs)) {
        i_lo <- pairs[1, k]; i_hi <- pairs[2, k]
        val_lo <- approx(curve_result$grid, curve_result$curve, xout = sv[i_lo], rule = 2)$y
        val_hi <- approx(curve_result$grid, curve_result$curve, xout = sv[i_hi], rule = 2)$y
        contrasts_df$contrast[k] <- paste0(sl[i_hi], " - ", sl[i_lo])
        contrasts_df$from_val[k] <- sv[i_lo]
        contrasts_df$to_val[k] <- sv[i_hi]
        contrasts_df$estimate[k] <- (val_hi - val_lo) / (sv[i_hi] - sv[i_lo])
      }
    }

    out <- list(
      variable = var,
      var_type = var_type,
      contrasts = contrasts_df,
      at = at, at_vals = at_vals, at_labels = at_labels,
      type = type, n_intervals = n_intervals, subset = subset
    )
    est_scalar <- contrasts_df$estimate[1]

  } else if (var_type == "categorical") {

    x_var_cat <- object$X[[var]]
    if (is.factor(x_var_cat)) {
      all_levels <- levels(x_var_cat)
    } else {
      all_levels <- sort(unique(as.character(x_var_cat)))
    }

    # Determine which contrasts: user-specified or all pairwise
    if (!is.null(at) && !identical(at, c(0.25, 0.75))) {
      at_levels <- as.character(at)
      bad <- setdiff(at_levels, all_levels)
      if (length(bad) > 0)
        stop(paste0("Levels not found in '", var, "': ", paste(bad, collapse = ", "),
                    ". Available: ", paste(all_levels, collapse = ", ")))
      if (length(at_levels) == 2) {
        pairs_list <- list(c(at_levels[1], at_levels[2]))
      } else {
        pairs_mat <- combn(at_levels, 2)
        pairs_list <- lapply(seq_len(ncol(pairs_mat)), function(k) pairs_mat[, k])
      }
    } else {
      # Default: all pairwise contrasts
      pairs_mat <- combn(all_levels, 2)
      pairs_list <- lapply(seq_len(ncol(pairs_mat)), function(k) pairs_mat[, k])
    }

    contrasts_df <- data.frame(
      contrast = character(length(pairs_list)),
      from_level = character(length(pairs_list)),
      to_level = character(length(pairs_list)),
      estimate = numeric(length(pairs_list)),
      se_sandwich = NA_real_,
      stringsAsFactors = FALSE
    )

    for (k in seq_along(pairs_list)) {
      lev_from <- pairs_list[[k]][1]
      lev_to   <- pairs_list[[k]][2]

      est_k <- .categorical_pairwise_effect(object, var, lev_to, lev_from,
                                            all_levels, subset = subset,
                                            compute_se = ci)
      contrasts_df$contrast[k]   <- paste0(lev_to, " - ", lev_from)
      contrasts_df$from_level[k] <- lev_from
      contrasts_df$to_level[k]   <- lev_to
      contrasts_df$estimate[k]   <- est_k$estimate
      if (ci && !is.null(est_k$se)) contrasts_df$se_sandwich[k] <- est_k$se
    }

    out <- list(
      variable = var,
      var_type = "categorical",
      contrasts = contrasts_df,
      levels = all_levels,
      n_intervals = 1L,
      subset = subset
    )
    est_scalar <- contrasts_df$estimate[1]

  } else {
    stop("Unrecognized variable type.")
  }

  # ============================================================
  # Variance estimation and CIs
  # ============================================================
  if (ci) {
    do_sandwich <- variance %in% c("sandwich", "both")
    do_pasr     <- variance %in% c("pasr", "both")
    z_crit <- qnorm(1 - alpha / 2)
    se_sand <- NULL; se_pasr <- NULL

    out$alpha <- alpha
    out$variance_method <- variance

    if (is_bin) {
      if (do_sandwich) {
        se_sand <- est$sandwich_se
        out$se_sandwich <- se_sand
        out$ci_lower_sandwich <- est_scalar - z_crit * se_sand
        out$ci_upper_sandwich <- est_scalar + z_crit * se_sand
      }
      if (do_pasr) {
        pasr_result <- .compute_pasr_se(
          object, var, at = at, type = type,
          bw = bw, q_lo = q_lo, q_hi = q_hi,
          subset = subset, ghat = ghat, is_bin = is_bin,
          R_min = R_min, R_max = R_max, batch_size = batch_size,
          tol = tol, n_stable = n_stable, B_mc = B_mc,
          nuisance = nuisance, verbose = verbose)
        se_pasr <- pasr_result$se
        out$se_pasr <- se_pasr
        out$C_psi <- pasr_result$C_psi
        out$V_psi <- pasr_result$V_psi
        out$R_used <- pasr_result$R_used
        out$converged <- pasr_result$converged
        out$ci_lower_pasr <- est_scalar - z_crit * se_pasr
        out$ci_upper_pasr <- est_scalar + z_crit * se_pasr
      }
      se_primary <- if (!is.null(se_pasr)) se_pasr else se_sand
      out$se <- se_primary
      out$ci_lower <- est_scalar - z_crit * se_primary
      out$ci_upper <- est_scalar + z_crit * se_primary

    } else if (var_type == "categorical") {
      # Categorical: SE already computed inline by .categorical_pairwise_effect
      n_contr <- nrow(out$contrasts)
      z_crit_cat <- qnorm(1 - alpha / 2)
      out$contrasts$ci_lower_sandwich <- out$contrasts$estimate - z_crit_cat * out$contrasts$se_sandwich
      out$contrasts$ci_upper_sandwich <- out$contrasts$estimate + z_crit_cat * out$contrasts$se_sandwich
      out$contrasts$se <- out$contrasts$se_sandwich
      out$contrasts$ci_lower <- out$contrasts$ci_lower_sandwich
      out$contrasts$ci_upper <- out$contrasts$ci_upper_sandwich
      out$se_sandwich <- out$contrasts$se_sandwich[1]
      out$se <- out$contrasts$se[1]

    } else {
      n_contr <- nrow(out$contrasts)
      out$contrasts$se_sandwich <- NA_real_
      out$contrasts$ci_lower_sandwich <- NA_real_
      out$contrasts$ci_upper_sandwich <- NA_real_
      out$contrasts$se_pasr <- NA_real_
      out$contrasts$ci_lower_pasr <- NA_real_
      out$contrasts$ci_upper_pasr <- NA_real_

      if (do_sandwich) {
        for (k in seq_len(n_contr)) {
          pair_at <- c(out$contrasts$from_val[k], out$contrasts$to_val[k])
          sand_k <- .compute_sandwich_se(object, var, at = pair_at, type = "value",
                                         bw = bw, q_lo = q_lo, q_hi = q_hi,
                                         subset = subset, ghat = ghat)
          out$contrasts$se_sandwich[k] <- sand_k$se
          out$contrasts$ci_lower_sandwich[k] <- out$contrasts$estimate[k] - z_crit * sand_k$se
          out$contrasts$ci_upper_sandwich[k] <- out$contrasts$estimate[k] + z_crit * sand_k$se
        }
        se_sand <- out$contrasts$se_sandwich[1]
        out$se_sandwich <- se_sand
      }

      if (do_pasr) {
        # PASR for primary contrast only (expensive)
        pair_at <- c(out$contrasts$from_val[1], out$contrasts$to_val[1])
        pasr_result <- .compute_pasr_se(
          object, var, at = pair_at, type = "value",
          bw = bw, q_lo = q_lo, q_hi = q_hi,
          subset = subset, ghat = ghat, is_bin = is_bin,
          R_min = R_min, R_max = R_max, batch_size = batch_size,
          tol = tol, n_stable = n_stable, B_mc = B_mc,
          nuisance = nuisance, verbose = verbose)
        se_pasr <- pasr_result$se
        out$se_pasr <- se_pasr
        out$C_psi <- pasr_result$C_psi
        out$V_psi <- pasr_result$V_psi
        out$R_used <- pasr_result$R_used
        out$converged <- pasr_result$converged
        out$contrasts$se_pasr[1] <- se_pasr
        out$contrasts$ci_lower_pasr[1] <- out$contrasts$estimate[1] - z_crit * se_pasr
        out$contrasts$ci_upper_pasr[1] <- out$contrasts$estimate[1] + z_crit * se_pasr
      }

      # Convenience aliases
      se_primary <- if (!is.null(se_pasr)) se_pasr else se_sand
      out$se <- se_primary
      out$contrasts$se <- NA_real_
      out$contrasts$ci_lower <- NA_real_
      out$contrasts$ci_upper <- NA_real_
      for (k in seq_len(n_contr)) {
        se_k <- if (!is.na(out$contrasts$se_pasr[k])) out$contrasts$se_pasr[k] else out$contrasts$se_sandwich[k]
        if (!is.na(se_k)) {
          out$contrasts$se[k] <- se_k
          out$contrasts$ci_lower[k] <- out$contrasts$estimate[k] - z_crit * se_k
          out$contrasts$ci_upper[k] <- out$contrasts$estimate[k] + z_crit * se_k
        }
      }
    }

    if (!is.null(se_pasr) && !is.null(se_sand))
      out$rho_V <- se_sand^2 / se_pasr^2
  }

  class(out) <- "infForest_effect"
  out
}


# ============================================================
# Categorical level effect: binary AIPW for one dummy (level vs not-level)
# ============================================================

#' Categorical pairwise effect: binary AIPW on the subset of observations
#' belonging to the two levels being compared. Returns estimate and
#' optionally the sandwich SE (computed from the same phi scores).
#' @keywords internal
.categorical_pairwise_effect <- function(object, var, lev_to, lev_from,
                                         all_levels, subset = NULL,
                                         compute_se = FALSE) {
  x_var_cat <- object$X[[var]]
  n <- nrow(object$X)

  # Subset to the two levels
  idx_to   <- which(as.character(x_var_cat) == lev_to)
  idx_from <- which(as.character(x_var_cat) == lev_from)
  idx_pair <- sort(c(idx_to, idx_from))
  if (!is.null(subset)) idx_pair <- intersect(idx_pair, subset)

  # Recode as binary: to=1, from=0
  x_binary <- rep(NA_real_, n)
  x_binary[idx_to]   <- 1
  x_binary[idx_from] <- 0

  # Propensity: P(to | X_{-race}) fitted on the pair subset only
  X_sub <- object$X[idx_pair, , drop = FALSE]
  X_sub[[var]] <- x_binary[idx_pair]
  prop <- .fit_propensity(X_sub, var, is_binary = TRUE)
  ghat <- rep(NA_real_, n)
  ghat[idx_pair] <- prop$ghat

  # Ranger integer codes for counterfactual routing
  if (is.factor(x_var_cat)) {
    code_to   <- as.numeric(factor(lev_to, levels = levels(x_var_cat)))
    code_from <- as.numeric(factor(lev_from, levels = levels(x_var_cat)))
  } else {
    code_to   <- as.numeric(factor(lev_to, levels = all_levels))
    code_from <- as.numeric(factor(lev_from, levels = all_levels))
  }

  # Collect phi scores for sandwich SE
  phi_sum <- numeric(n)
  phi_cnt <- integer(n)

  psi_splits <- numeric(object$honesty.splits)
  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]

    # Use original X_ord (no modification — routing needs correct factor codes)
    X_ord_A <- .get_X_ord(object, fs$rfA)
    col_idx_A <- get_ranger_col_idx(fs$rfA, var)
    X_ord_B <- .get_X_ord(object, fs$rfB)
    col_idx_B <- get_ranger_col_idx(fs$rfB, var)

    # Restrict honest obs to the pair subset
    hon_B <- intersect(fs$idxB, idx_pair)
    hon_A <- intersect(fs$idxA, idx_pair)

    y_hon_AB <- rep(NA_real_, n)
    y_hon_AB[hon_B] <- as.numeric(object$Y[hon_B])
    res_AB <- aipw_scores_v2_cpp(fs$rfA$forest, X_ord_A, y_hon_AB,
                                 as.integer(hon_B), ghat,
                                 col_idx_A, TRUE, code_to, code_from,
                                 indicator_ = x_binary)

    y_hon_BA <- rep(NA_real_, n)
    y_hon_BA[hon_A] <- as.numeric(object$Y[hon_A])
    res_BA <- aipw_scores_v2_cpp(fs$rfB$forest, X_ord_B, y_hon_BA,
                                 as.integer(hon_A), ghat,
                                 col_idx_B, TRUE, code_to, code_from,
                                 indicator_ = x_binary)

    psi_splits[r] <- (res_AB$psi + res_BA$psi) / 2

    # Accumulate phi scores for sandwich
    if (compute_se) {
      for (j in seq_along(hon_B)) {
        k <- hon_B[j]
        if (!is.na(res_AB$phi[j])) {
          phi_sum[k] <- phi_sum[k] + res_AB$phi[j]
          phi_cnt[k] <- phi_cnt[k] + 1L
        }
      }
      for (j in seq_along(hon_A)) {
        k <- hon_A[j]
        if (!is.na(res_BA$phi[j])) {
          phi_sum[k] <- phi_sum[k] + res_BA$phi[j]
          phi_cnt[k] <- phi_cnt[k] + 1L
        }
      }
    }
  }

  estimate <- mean(psi_splits)

  if (!compute_se) return(list(estimate = estimate))

  # Sandwich SE from phi scores
  valid <- phi_cnt > 0
  n_valid <- sum(valid)
  phi_avg <- rep(NA_real_, n)
  phi_avg[valid] <- phi_sum[valid] / phi_cnt[valid]

  psi_bar <- mean(phi_avg[valid])
  V_IF <- sum((phi_avg[valid] - psi_bar)^2) / (n_valid * (n_valid - 1))

  list(estimate = estimate, se = sqrt(V_IF))
}


# ============================================================
# AIPW internals
# ============================================================

#' Get ranger-ordered X matrix, using cache on infForest object if available
#' @keywords internal
.get_X_ord <- function(object, rf) {
  if (!is.null(object$X_ord) && is.numeric(object$X_ord)) return(object$X_ord)
  # Convert factors to integer codes before as.matrix (ranger uses integer encoding)
  X_df <- object$X
  for (col in names(X_df)) {
    if (is.factor(X_df[[col]]) || is.character(X_df[[col]])) {
      X_df[[col]] <- as.numeric(as.factor(X_df[[col]]))
    }
  }
  vnames <- rf$forest$independent.variable.names
  as.matrix(X_df[, vnames])
}

#' Fit propensity model: predict X_j from X_{-j} using LOO ridge
#'
#' For continuous X_j: ridge regression with leave-one-out predictions via
#' the PRESS formula. LOO ensures Cov(ghat, e_j) = 0, which eliminates the
#' propensity attenuation factor (lambda = 1 exactly).
#'
#' For binary X_j: logistic ridge with K-fold cross-validated predictions.
#' Observation k's prediction never uses observation k in the fit.
#'
#' @keywords internal
.fit_propensity <- function(X, var, is_binary, n_trees = NULL) {
  var_col_r <- which(names(X) == var)
  X_minus <- X[, -var_col_r, drop = FALSE]
  # Convert factors/characters to integer codes for glmnet
  for (col in names(X_minus)) {
    if (is.factor(X_minus[[col]]) || is.character(X_minus[[col]])) {
      X_minus[[col]] <- as.numeric(as.factor(X_minus[[col]]))
    }
  }
  Xm <- as.matrix(X_minus)
  x_j <- X[[var]]
  n <- length(x_j)
  p <- ncol(Xm)

  if (is_binary) {
    # Binary: K-fold logistic ridge, predictions are out-of-fold
    K <- 10L
    cv_fit <- glmnet::cv.glmnet(Xm, x_j, alpha = 0,
                                family = "binomial", nfolds = 5)
    lambda_use <- cv_fit$lambda.min

    folds <- sample(rep(seq_len(K), length.out = n))
    ghat <- numeric(n)
    for (fold in seq_len(K)) {
      train <- which(folds != fold)
      test  <- which(folds == fold)
      fit_k <- glmnet::glmnet(Xm[train, , drop = FALSE], x_j[train],
                              alpha = 0, family = "binomial",
                              lambda = lambda_use)
      ghat[test] <- as.numeric(predict(fit_k, Xm[test, , drop = FALSE],
                                       type = "response"))
    }
    ghat <- pmax(pmin(ghat, 0.975), 0.025)

  } else {
    # Continuous: ridge LOO via PRESS formula
    # ghat_LOO(k) = x_j(k) - e_insample(k) / (1 - h_kk)
    cv_fit <- glmnet::cv.glmnet(Xm, x_j, alpha = 0, nfolds = 5)
    lambda_use <- cv_fit$lambda.min

    ghat_insample <- as.numeric(predict(cv_fit, Xm, s = lambda_use))
    e_insample <- x_j - ghat_insample

    # Hat matrix diagonal: h_kk = diag(X (X'X + lambda I)^{-1} X')
    XtX_ridge_inv <- solve(crossprod(Xm) + lambda_use * diag(p))
    H_diag <- rowSums((Xm %*% XtX_ridge_inv) * Xm)

    # LOO predictions
    e_loo <- e_insample / (1 - H_diag)
    ghat <- x_j - e_loo
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
  n <- nrow(object$X)
  psi_splits <- numeric(object$honesty.splits)
  diag_list <- vector("list", object$honesty.splits)

  if (is.null(ghat)) {
    prop <- .fit_propensity(object$X, var, is_binary = TRUE,
                            n_trees = propensity_trees)
    ghat <- prop$ghat
  }

  # Collect phi scores for sandwich SE in the same pass
  phi_sum <- numeric(n)
  phi_cnt <- integer(n)

  has_cache <- !is.null(object$forest_caches)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]

    if (has_cache && is.null(subset)) {
      col_idx_A <- get_ranger_col_idx(fs$rfA, var)
      cache_AB <- object$forest_caches[[paste0(r, "_AB")]]
      res_AB <- aipw_scores_cached_cpp(cache_AB, ghat, col_idx_A, TRUE, 1, 0)
      col_idx_B <- get_ranger_col_idx(fs$rfB, var)
      cache_BA <- object$forest_caches[[paste0(r, "_BA")]]
      res_BA <- aipw_scores_cached_cpp(cache_BA, ghat, col_idx_B, TRUE, 1, 0)
    } else {
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
    }

    psi_splits[r] <- (res_AB$psi + res_BA$psi) / 2
    diag_list[[r]] <- list(AB = res_AB, BA = res_BA)

    # Accumulate phi scores for sandwich
    hon_B <- if (!is.null(subset)) intersect(fs$idxB, subset) else fs$idxB
    hon_A <- if (!is.null(subset)) intersect(fs$idxA, subset) else fs$idxA
    for (j in seq_along(hon_B)) {
      k <- hon_B[j]
      if (!is.na(res_AB$phi[j])) { phi_sum[k] <- phi_sum[k] + res_AB$phi[j]; phi_cnt[k] <- phi_cnt[k] + 1L }
    }
    for (j in seq_along(hon_A)) {
      k <- hon_A[j]
      if (!is.na(res_BA$phi[j])) { phi_sum[k] <- phi_sum[k] + res_BA$phi[j]; phi_cnt[k] <- phi_cnt[k] + 1L }
    }
  }

  # Compute sandwich SE from accumulated phi
  valid <- phi_cnt > 0
  n_valid <- sum(valid)
  phi_avg <- rep(NA_real_, n)
  phi_avg[valid] <- phi_sum[valid] / phi_cnt[valid]
  psi_bar <- mean(phi_avg[valid])
  V_IF <- sum((phi_avg[valid] - psi_bar)^2) / (n_valid * (n_valid - 1))

  list(
    psi = mean(psi_splits),
    per_split = psi_splits,
    diagnostics = diag_list,
    sandwich_se = sqrt(V_IF)
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

  pct <- round((1 - (if (!is.null(x$alpha)) x$alpha else 0.05)) * 100)

  if (x$var_type == "binary") {
    cat(sprintf("  Estimate:    %.4f\n", x$estimate))
    if (!is.null(x$se_sandwich)) {
      cat(sprintf("  SE (sandwich): %.4f  |  %d%% CI: [%.4f, %.4f]\n",
                  x$se_sandwich, pct, x$ci_lower_sandwich, x$ci_upper_sandwich))
    }
    if (!is.null(x$se_pasr)) {
      cat(sprintf("  SE (PASR):     %.4f  |  %d%% CI: [%.4f, %.4f]\n",
                  x$se_pasr, pct, x$ci_lower_pasr, x$ci_upper_pasr))
    }
    if (!is.null(x$rho_V))
      cat(sprintf("  rho_V:         %.2f\n", x$rho_V))
  } else {
    if (x$var_type == "continuous") {
      cat("  Intervals:  ", x$n_intervals, "\n\n")
      cat("  Contrasts (per unit):\n")
    } else {
      cat("\n  Contrasts:\n")
    }
    df <- x$contrasts
    for (k in seq_len(nrow(df))) {
      if (x$var_type == "continuous") {
        cat(sprintf("    %-16s [%.3f, %.3f]:  %.4f\n",
                    df$contrast[k], df$from_val[k], df$to_val[k], df$estimate[k]))
      } else {
        cat(sprintf("    %-25s:  %8.4f\n", df$contrast[k], df$estimate[k]))
      }
      if ("se_sandwich" %in% names(df) && !is.na(df$se_sandwich[k]))
        cat(sprintf("      SE (sandwich): %.4f  |  %d%% CI: [%.4f, %.4f]\n",
                    df$se_sandwich[k], pct, df$ci_lower_sandwich[k], df$ci_upper_sandwich[k]))
      if ("se_pasr" %in% names(df) && !is.na(df$se_pasr[k]))
        cat(sprintf("      SE (PASR):     %.4f  |  %d%% CI: [%.4f, %.4f]\n",
                    df$se_pasr[k], pct, df$ci_lower_pasr[k], df$ci_upper_pasr[k]))
    }
    if (!is.null(x$rho_V))
      cat(sprintf("\n  rho_V: %.2f\n", x$rho_V))
  }
  invisible(x)
}
