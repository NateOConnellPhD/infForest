# ============================================================
# Variance estimation internals for infForest
#
# Contains:
#   .compute_sandwich_se()   — sandwich SE from AIPW influence function scores
#   .compute_pasr_se()       — PASR SE for marginal effects
#   .compute_pasr_int_se()   — PASR SE for interaction differences
#
# Called by effect() and int() when ci = TRUE.
# ============================================================

# ============================================================
# Sandwich (influence function) variance estimator
# Called by effect() and int() when ci = TRUE
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

  phi_sum <- numeric(n)
  phi_cnt <- integer(n)

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

    if (is_bin) { a_sc <- 1; b_sc <- 0
    } else { a_sc <- a_val; b_sc <- b_val }

    y_hon_AB <- rep(NA_real_, n); y_hon_AB[hon_B] <- as.numeric(Y[hon_B])
    res_AB <- aipw_scores_v2_cpp(fs$rfA$forest, X_ord_A, y_hon_AB,
                                 as.integer(hon_B), ghat,
                                 col_idx_A, is_bin, a_sc, b_sc)
    for (j in seq_along(hon_B)) {
      k <- hon_B[j]
      if (!is.na(res_AB$phi[j])) {
        phi_sum[k] <- phi_sum[k] + res_AB$phi[j]
        phi_cnt[k] <- phi_cnt[k] + 1L
      }
    }

    y_hon_BA <- rep(NA_real_, n); y_hon_BA[hon_A] <- as.numeric(Y[hon_A])
    res_BA <- aipw_scores_v2_cpp(fs$rfB$forest, X_ord_B, y_hon_BA,
                                 as.integer(hon_A), ghat,
                                 col_idx_B, is_bin, a_sc, b_sc)
    for (j in seq_along(hon_A)) {
      k <- hon_A[j]
      if (!is.na(res_BA$phi[j])) {
        phi_sum[k] <- phi_sum[k] + res_BA$phi[j]
        phi_cnt[k] <- phi_cnt[k] + 1L
      }
    }
  }

  valid <- phi_cnt > 0
  n_valid <- sum(valid)
  phi_avg <- rep(NA_real_, n)
  phi_avg[valid] <- phi_sum[valid] / phi_cnt[valid]

  psi_bar <- mean(phi_avg[valid])
  V_IF <- sum((phi_avg[valid] - psi_bar)^2) / (n_valid * (n_valid - 1))

  list(se = sqrt(V_IF), V_IF = V_IF, n_valid = n_valid, psi_bar = psi_bar)
}


# ============================================================
# PASR variance estimator for marginal effects
# ============================================================

#' @keywords internal
.compute_pasr_se <- function(object, var, at = c(0.25, 0.75),
                             type = "quantile", bw = 20L,
                             q_lo = 0.10, q_hi = 0.90,
                             subset = NULL, ghat = NULL, is_bin = FALSE,
                             R_min = 20L, R_max = 200L,
                             batch_size = 10L, tol = 0.05,
                             n_stable = 2L, B_mc = 500L,
                             nuisance = NULL, verbose = FALSE) {

  X <- object$X; Y <- object$Y; n <- nrow(X)
  if (is.null(nuisance)) nuisance <- estimate_nuisance(object)

  # V_psi / R from honesty splits
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

  # Pre-compute shared objects
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

  .build_rf_args_local <- function(dat, seed_val) {
    args <- list(formula = y ~ ., data = dat,
                 num.trees = object$params$num.trees,
                 mtry = object$params$mtry,
                 min.node.size = object$params$min.node.size,
                 sample.fraction = object$params$sample.fraction,
                 replace = object$params$replace,
                 num.threads = 1L,
                 write.forest = TRUE, seed = seed_val,
                 penalize.split.competition = object$params$penalize,
                 softmax.split = object$params$softmax)
    if (object$outcome_type == "binary") args$probability <- TRUE
    args
  }

  .pasr_extract_local <- function(rf_fA, rf_fB, idxA, idxB, Y_syn) {
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
      rfA1 <- do.call(inf.ranger::ranger, .build_rf_args_local(dat_fA, r * 100L + 1L))
      rfA2 <- do.call(inf.ranger::ranger, .build_rf_args_local(dat_fB, r * 100L + 1L))
      rfB1 <- do.call(inf.ranger::ranger, .build_rf_args_local(dat_fA, r * 100L + 2L))
      rfB2 <- do.call(inf.ranger::ranger, .build_rf_args_local(dat_fB, r * 100L + 2L))
      psi_A[r] <- .pasr_extract_local(rfA1, rfA2, idxA, idxB, Y_syn)
      psi_B[r] <- .pasr_extract_local(rfB1, rfB2, idxA, idxB, Y_syn)
    }
    if (R_current >= R_min) {
      pA <- psi_A[1:R_current]; pB <- psi_B[1:R_current]
      C_current <- max(cov(pA, pB), 0)
      rel_change <- if (is.finite(C_prev)) abs(C_current - C_prev) / max(C_prev, 1e-10) else Inf
      if (verbose)
        cat(sprintf("  PASR R=%d: C_psi=%.6f  rel_change=%.4f  stable=%d/%d\n",
                    R_current, C_current, rel_change, stable_count, n_stable))
      if (rel_change < tol) stable_count <- stable_count + 1L else stable_count <- 0L
      if (stable_count >= n_stable) { converged <- TRUE; break }
      C_prev <- C_current
    }
  }

  pA <- psi_A[1:R_current]; pB <- psi_B[1:R_current]
  C_psi <- max(cov(pA, pB), 0)
  total_var <- V_psi_over_R + C_psi

  list(se = sqrt(total_var), C_psi = C_psi, V_psi = V_psi_over_R,
       R_used = R_current, converged = converged)
}

# ============================================================
# PASR variance for interaction differences
# ============================================================

#' @keywords internal
.compute_pasr_int_se <- function(object, var, by,
                                 at = c(0.25, 0.75), type = "quantile",
                                 bw = 20L, q_lo = 0.10, q_hi = 0.90,
                                 subset = NULL,
                                 R_min = 20L, R_max = 200L,
                                 batch_size = 10L, tol = 0.05,
                                 n_stable = 2L, B_mc = 500L,
                                 nuisance = NULL, verbose = FALSE) {

  X <- object$X; n <- nrow(X)
  if (is.null(nuisance)) nuisance <- estimate_nuisance(object)

  psi_A <- numeric(R_max); psi_B <- numeric(R_max)
  R_current <- 0L; C_prev <- Inf; stable_count <- 0L; converged <- FALSE

  while (R_current < R_max) {
    for (bi in seq_len(batch_size)) {
      R_current <- R_current + 1L; r <- R_current
      Y_syn <- generate_synthetic_Y(nuisance, seed = r * 7919L)
      dat_syn <- X
      if (object$outcome_type == "continuous") { dat_syn$y <- Y_syn
      } else { dat_syn$y <- factor(Y_syn, levels = c(0, 1)) }

      fitA <- tryCatch(
        infForest(y ~ ., data = dat_syn, num.trees = object$params$num.trees,
                  mtry = object$params$mtry,
                  min.node.size = object$params$min.node.size,
                  sample.fraction = object$params$sample.fraction,
                  replace = object$params$replace,
                  penalize = object$params$penalize,
                  softmax = object$params$softmax,
                  seed = r * 2L - 1L, honesty.splits = 1L),
        error = function(e) NULL)
      fitB <- tryCatch(
        infForest(y ~ ., data = dat_syn, num.trees = object$params$num.trees,
                  mtry = object$params$mtry,
                  min.node.size = object$params$min.node.size,
                  sample.fraction = object$params$sample.fraction,
                  replace = object$params$replace,
                  penalize = object$params$penalize,
                  softmax = object$params$softmax,
                  seed = r * 2L, honesty.splits = 1L),
        error = function(e) NULL)

      intA <- if (!is.null(fitA)) tryCatch(
        int(fitA, var, by = by, at = at, type = type,
            bw = bw, q_lo = q_lo, q_hi = q_hi, subset = subset, ci = FALSE),
        error = function(e) NULL) else NULL
      intB <- if (!is.null(fitB)) tryCatch(
        int(fitB, var, by = by, at = at, type = type,
            bw = bw, q_lo = q_lo, q_hi = q_hi, subset = subset, ci = FALSE),
        error = function(e) NULL) else NULL

      psi_A[r] <- if (!is.null(intA)) intA$differences$difference[1] else NA_real_
      psi_B[r] <- if (!is.null(intB)) intB$differences$difference[1] else NA_real_
    }

    if (R_current >= R_min) {
      pA <- psi_A[1:R_current]; pB <- psi_B[1:R_current]
      valid <- !is.na(pA) & !is.na(pB)
      if (sum(valid) >= 4) {
        C_current <- max(cov(pA[valid], pB[valid]), 0)
        rel_change <- if (is.finite(C_prev) && C_prev > 0)
          abs(C_current - C_prev) / C_prev else Inf
        if (verbose)
          cat(sprintf("  PASR int R=%d: C_psi=%.6f  rel_change=%.4f  stable=%d/%d\n",
                      sum(valid), C_current, rel_change, stable_count, n_stable))
        if (rel_change < tol) stable_count <- stable_count + 1L else stable_count <- 0L
        if (stable_count >= n_stable) { converged <- TRUE; break }
        C_prev <- C_current
      }
    }
  }

  pA <- psi_A[1:R_current]; pB <- psi_B[1:R_current]
  valid <- !is.na(pA) & !is.na(pB)
  C_psi <- if (sum(valid) >= 4) max(cov(pA[valid], pB[valid]), 0) else 0

  list(se = sqrt(C_psi), C_psi = C_psi, R_used = sum(valid), converged = converged)
}
