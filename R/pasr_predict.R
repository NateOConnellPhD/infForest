#' PASR Prediction Intervals for Ranger Forests
#'
#' Estimates pointwise prediction intervals (continuous) or confidence intervals
#' for predicted probabilities (binary) from a fitted \code{ranger} model using
#' Procedure-Aligned Synthetic Resampling (PASR).
#'
#' For continuous outcomes, returns both confidence intervals for E[Y|X=x]
#' and prediction intervals for Y_new|X=x. For binary outcomes, returns
#' confidence intervals for P(Y=1|X=x). See O'Connell (2026) for theory.
#'
#' This function is for pointwise predictions only. It has nothing to do with
#' effect estimation or inference — use \code{effect()}, \code{int()}, and
#' \code{summary()} from an \code{infForest} object for that.
#'
#' @param object A fitted \code{ranger} object.
#' @param data Training data frame used to fit the ranger model. Required
#'   because ranger does not store training data.
#' @param newdata Data frame of prediction points. Default \code{NULL} uses
#'   training data.
#' @param R_min Minimum PASR replicates before checking convergence. Default 20.
#' @param R_max Maximum PASR replicates. Default 200.
#' @param batch_size Replicates per convergence check. Default 10.
#' @param tol Relative change tolerance for convergence. Default 0.05.
#' @param n_stable Consecutive stable batches required. Default 2.
#' @param B_mc Trees per paired forest in PASR. Default 500.
#' @param alpha Significance level. Default 0.05.
#' @param B_loc Trees for cross-fitted mean forests (continuous only). Default 1000.
#' @param B_scale Trees for variance forest (continuous only). Default 1500.
#' @param R_cf Cross-fitting repetitions (continuous only). Default 5.
#' @param verbose Print convergence progress. Default FALSE.
#' @param ... Additional arguments (unused).
#'
#' @return A data frame with columns: f_hat, se, ci_lower, ci_upper, mc_var,
#'   Ct_hat, R_used, converged. For continuous outcomes, also: pi_lower,
#'   pi_upper, sigma2_hat.
#'
#' @export
pasr_predict <- function(object, data, newdata = NULL,
                         R_min = 20L, R_max = 200L, batch_size = 10L,
                         tol = 0.05, n_stable = 2L,
                         B_mc = 500L, alpha = 0.05,
                         B_loc = 1000L, B_scale = 1500L, R_cf = 5L,
                         verbose = FALSE, ...) {

  if (!inherits(object, "ranger")) stop("pasr_predict requires a fitted ranger object.")
  if (missing(data)) stop("'data' (training data frame) is required.")

  z_crit <- qnorm(1 - alpha / 2)

  # Detect outcome type
  resp_name <- setdiff(names(data), object$forest$independent.variable.names)
  if (length(resp_name) != 1) stop("Cannot identify response column in data.")
  Y_raw <- data[[resp_name]]
  is_prob_forest <- !is.null(object$treetype) && object$treetype == "Probability estimation"
  if ((is.factor(Y_raw) && nlevels(Y_raw) == 2) || is_prob_forest) {
    outcome_type <- "binary"
    Y <- if (is.factor(Y_raw)) as.numeric(Y_raw) - 1 else as.numeric(Y_raw)
  } else {
    outcome_type <- "continuous"
    Y <- as.numeric(Y_raw)
  }

  X <- data[, object$forest$independent.variable.names, drop = FALSE]
  n <- nrow(X)
  p <- ncol(X)
  vn <- object$forest$independent.variable.names

  if (is.null(newdata)) newdata <- X
  nk <- nrow(newdata)

  # ===========================================================
  # Deployed forest predictions and within-forest MC variance
  # ===========================================================
  pred_all <- predict(object, data = newdata[, vn, drop = FALSE],
                      predict.all = TRUE)$predictions
  if (outcome_type == "binary") {
    pred_all <- pred_all[, 2, ]
  }

  f_hat <- rowMeans(pred_all)
  if (outcome_type == "binary") f_hat <- pmin(pmax(f_hat, 1e-4), 1 - 1e-4)
  s2_trees <- rowSums((pred_all - f_hat)^2) / (ncol(pred_all) - 1)
  mc_var <- s2_trees / ncol(pred_all)

  # ===========================================================
  # Nuisance estimation: continuous only
  # Cross-fitted residual product (Section 7.3, Equation 6)
  # No nuisance for binary — Bernoulli variance is implicit
  # ===========================================================
  sigma2_hat <- NULL
  mhat <- NULL
  sigma_hat_train <- NULL

  if (outcome_type == "continuous") {

    .fit_mean_pred <- function(idx_tr, idx_te, seed) {
      dat_tr <- X[idx_tr, , drop = FALSE]
      dat_tr$y <- Y[idx_tr]
      rf <- inf.ranger::ranger(y ~ ., data = dat_tr, num.trees = B_loc,
                                mtry = p, min.node.size = 1,
                                sample.fraction = 1.0, replace = FALSE,
                                seed = seed, num.threads = 1L)
      as.numeric(predict(rf, data = X[idx_te, , drop = FALSE])$predictions)
    }

    # Two independent cross-fitted mean sequences
    m1_sum <- numeric(n); m1_cnt <- integer(n)
    m2_sum <- numeric(n); m2_cnt <- integer(n)

    for (r in seq_len(R_cf)) {
      idxA <- sample.int(n, n %/% 2)
      idxB <- setdiff(seq_len(n), idxA)
      m1_sum[idxB] <- m1_sum[idxB] + .fit_mean_pred(idxA, idxB, r * 100L + 1L)
      m1_sum[idxA] <- m1_sum[idxA] + .fit_mean_pred(idxB, idxA, r * 100L + 2L)
      m1_cnt[idxB] <- m1_cnt[idxB] + 1L
      m1_cnt[idxA] <- m1_cnt[idxA] + 1L

      idxC <- sample.int(n, n %/% 2)
      idxD <- setdiff(seq_len(n), idxC)
      m2_sum[idxD] <- m2_sum[idxD] + .fit_mean_pred(idxC, idxD, r * 100L + 3L)
      m2_sum[idxC] <- m2_sum[idxC] + .fit_mean_pred(idxD, idxC, r * 100L + 4L)
      m2_cnt[idxD] <- m2_cnt[idxD] + 1L
      m2_cnt[idxC] <- m2_cnt[idxC] + 1L
    }

    mhat1 <- m1_sum / pmax(m1_cnt, 1L)
    mhat2 <- m2_sum / pmax(m2_cnt, 1L)

    # Cross-fitted residual product (Eq 6)
    sprod <- (Y - mhat1) * (Y - mhat2)

    # Variance forest on the residual product
    dat_scale <- X
    dat_scale$sprod <- sprod
    rf_scale <- inf.ranger::ranger(sprod ~ ., data = dat_scale, num.trees = B_scale,
                                    mtry = p, min.node.size = object$min.node.size,
                                    sample.fraction = 1.0, replace = FALSE,
                                    num.threads = 1L)

    sigma2_hat <- pmax(as.numeric(predict(rf_scale, data = newdata[, vn, drop = FALSE])$predictions), 1e-8)

    # For synthetic Y generation
    mhat <- (mhat1 + mhat2) / 2
    sigma_hat_train <- sqrt(pmax(as.numeric(predict(rf_scale, data = X)$predictions), 1e-8))

    if (verbose) cat("  Nuisance: cross-fitted residual product complete\n")
  }

  # ===========================================================
  # Binary: fitted probabilities for synthetic Y
  # ===========================================================
  if (outcome_type == "binary") {
    pred_train <- predict(object, data = X[, vn, drop = FALSE])$predictions
    if (is.matrix(pred_train)) pred_train <- pred_train[, 2]
    pred_train <- pmin(pmax(pred_train, 0.001), 0.999)
  }

  # ===========================================================
  # PASR: paired forests on synthetic replicates
  # ===========================================================
  psi_A_mat <- matrix(NA, nk, 0)
  psi_B_mat <- matrix(NA, nk, 0)
  R_current <- 0L
  Ct_prev <- rep(Inf, nk)
  stable_count <- 0L
  converged <- FALSE

  rf_args_base <- list(
    formula = as.formula(paste(resp_name, "~ .")),
    num.trees = B_mc,
    mtry = object$mtry,
    min.node.size = object$min.node.size,
    sample.fraction = 1.0, replace = FALSE,
    num.threads = 1L, write.forest = TRUE
  )
  if (outcome_type == "binary") rf_args_base$probability <- TRUE

  while (R_current < R_max) {
    batch_A <- matrix(NA, nk, batch_size)
    batch_B <- matrix(NA, nk, batch_size)

    for (b in seq_len(batch_size)) {
      R_current <- R_current + 1L
      r <- R_current

      set.seed(r * 7919L)
      if (outcome_type == "continuous") {
        Y_syn <- mhat + sigma_hat_train * rnorm(n)
      } else {
        Y_syn <- factor(rbinom(n, 1, pred_train), levels = c(0, 1))
      }

      dat_syn <- X
      dat_syn[[resp_name]] <- Y_syn

      rf_args <- rf_args_base
      rf_args$data <- dat_syn

      rf_args$seed <- r * 100L + 1L
      rfA_syn <- do.call(inf.ranger::ranger, rf_args)

      rf_args$seed <- r * 100L + 2L
      rfB_syn <- do.call(inf.ranger::ranger, rf_args)

      pA <- predict(rfA_syn, data = newdata[, vn, drop = FALSE])$predictions
      pB <- predict(rfB_syn, data = newdata[, vn, drop = FALSE])$predictions

      if (outcome_type == "binary") { pA <- pA[, 2]; pB <- pB[, 2] }

      batch_A[, b] <- pA
      batch_B[, b] <- pB
    }

    psi_A_mat <- cbind(psi_A_mat, batch_A)
    psi_B_mat <- cbind(psi_B_mat, batch_B)

    if (R_current >= R_min) {
      Ct_current <- numeric(nk)
      for (k in seq_len(nk)) {
        Ct_current[k] <- max(cov(psi_A_mat[k, ], psi_B_mat[k, ]), 0)
      }

      rel_changes <- ifelse(is.finite(Ct_prev),
                            abs(Ct_current - Ct_prev) / pmax(Ct_prev, 1e-10),
                            Inf)
      med_rel_change <- median(rel_changes)

      if (verbose) {
        cat(sprintf("  PASR R=%d: median_Ct=%.6f  med_rel_change=%.4f  stable=%d/%d\n",
                    R_current, median(Ct_current), med_rel_change, stable_count, n_stable))
      }

      if (med_rel_change < tol) {
        stable_count <- stable_count + 1L
      } else {
        stable_count <- 0L
      }

      if (stable_count >= n_stable) {
        converged <- TRUE
        break
      }
      Ct_prev <- Ct_current
    }
  }

  # Final Ct
  Ct_hat <- numeric(nk)
  for (k in seq_len(nk)) {
    Ct_hat[k] <- max(cov(psi_A_mat[k, ], psi_B_mat[k, ]), 0)
  }

  if (verbose) {
    if (converged) {
      cat(sprintf("  PASR converged at R=%d (median Ct=%.6f)\n", R_current, median(Ct_hat)))
    } else {
      cat(sprintf("  PASR reached R_max=%d without convergence (median Ct=%.6f)\n", R_max, median(Ct_hat)))
    }
  }

  # ===========================================================
  # Build intervals (Equations 9 and 10 from the paper)
  # ===========================================================

  # CI: mc_var + Ct (both outcome types)
  ci_var <- mc_var + Ct_hat
  ci_se <- sqrt(ci_var)

  out <- data.frame(
    f_hat = f_hat,
    se = ci_se,
    ci_lower = f_hat - z_crit * ci_se,
    ci_upper = f_hat + z_crit * ci_se,
    mc_var = mc_var,
    Ct_hat = Ct_hat,
    R_used = R_current,
    converged = converged
  )

  # PI: sigma2 + mc_var + Ct (continuous only, Equation 9)
  if (outcome_type == "continuous" && !is.null(sigma2_hat)) {
    pi_var <- sigma2_hat + mc_var + Ct_hat
    pi_se <- sqrt(pi_var)
    out$pi_lower <- f_hat - z_crit * pi_se
    out$pi_upper <- f_hat + z_crit * pi_se
    out$sigma2_hat <- sigma2_hat
  }

  out
}
