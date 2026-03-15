#' PASR for Pointwise Predictions
#'
#' Estimates the covariance floor C_T(x) for pointwise forest predictions using
#' procedure-aligned synthetic resampling with adaptive convergence. Returns
#' confidence intervals for f_0(x) and prediction intervals for Y_new
#' (continuous outcomes only).
#'
#' @param object An \code{infForest} object.
#' @param newdata Data frame of query points. Default \code{NULL} uses training data.
#' @param R_min Minimum replicates before checking convergence. Default 20.
#' @param R_max Maximum replicates (hard cap). Default 200.
#' @param batch_size Replicates per convergence check. Default 10.
#' @param tol Relative change tolerance for convergence. Default 0.05.
#' @param n_stable Consecutive stable batches required. Default 2.
#' @param B_mc Number of trees per paired forest. Default 500.
#' @param alpha Significance level. Default 0.05.
#' @param nuisance An \code{infForest_nuisance} object. If \code{NULL}, estimated automatically.
#' @param verbose Print convergence progress. Default \code{FALSE}.
#' @param ... Additional arguments passed to \code{estimate_nuisance}.
#'
#' @return A data frame with columns f_hat, se, ci_lower, ci_upper, mc_var,
#'   Ct_hat, R_used, converged. For continuous outcomes, also pi_lower,
#'   pi_upper, sigma2_hat.
#'
#' @export
pasr_predict <- function(object, newdata = NULL,
                         R_min = 20L, R_max = 200L, batch_size = 10L,
                         tol = 0.05, n_stable = 2L,
                         B_mc = 500L, alpha = 0.05,
                         nuisance = NULL, verbose = FALSE, ...) {

  check_infForest(object)
  z_crit <- qnorm(1 - alpha / 2)

  X <- object$X
  Y <- object$Y
  n <- nrow(X)
  p <- ncol(X)

  if (is.null(newdata)) newdata <- X
  nk <- nrow(newdata)

  # --- Estimate nuisance if not provided ---
  if (is.null(nuisance)) {
    nuisance <- estimate_nuisance(object, ...)
  }

  # --- Deployed forest predictions and MC variance ---
  fs <- object$forests[[1]]
  vn <- fs$rfA$forest$independent.variable.names

  predA_all <- predict(fs$rfA, data = newdata[, vn, drop = FALSE],
                       predict.all = TRUE)$predictions
  if (object$outcome_type == "binary") {
    predA_all <- predA_all[, 2, ]
  }
  f_hatA <- rowMeans(predA_all)
  s2A <- rowSums((predA_all - f_hatA)^2) / (ncol(predA_all) - 1)
  mc_varA <- s2A / ncol(predA_all)

  predB_all <- predict(fs$rfB, data = newdata[, vn, drop = FALSE],
                       predict.all = TRUE)$predictions
  if (object$outcome_type == "binary") {
    predB_all <- predB_all[, 2, ]
  }
  f_hatB <- rowMeans(predB_all)
  s2B <- rowSums((predB_all - f_hatB)^2) / (ncol(predB_all) - 1)
  mc_varB <- s2B / ncol(predB_all)

  f_hat <- (f_hatA + f_hatB) / 2
  mc_var <- (mc_varA + mc_varB) / 4

  # --- PASR with convergence ---
  psi_A_mat <- matrix(NA, nk, 0)
  psi_B_mat <- matrix(NA, nk, 0)
  R_current <- 0L
  Ct_prev <- rep(Inf, nk)
  stable_count <- 0L
  converged <- FALSE

  while (R_current < R_max) {
    batch_A <- matrix(NA, nk, batch_size)
    batch_B <- matrix(NA, nk, batch_size)

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

      rf_args <- list(formula = y ~ ., data = dat_syn, num.trees = B_mc,
                      mtry = object$params$mtry, min.node.size = object$params$min.node.size,
                      sample.fraction = 1.0, replace = FALSE,
                      num.threads = 1L, write.forest = TRUE,
                      penalize.split.competition = object$params$penalize,
                      softmax.split = object$params$softmax)
      if (object$outcome_type == "binary") rf_args$probability <- TRUE

      rf_args$seed <- r * 100L + 1L
      rfA_syn <- do.call(inf.ranger::ranger, rf_args)

      rf_args$seed <- r * 100L + 2L
      rfB_syn <- do.call(inf.ranger::ranger, rf_args)

      pA <- predict(rfA_syn, data = newdata[, vn, drop = FALSE])$predictions
      pB <- predict(rfB_syn, data = newdata[, vn, drop = FALSE])$predictions

      if (object$outcome_type == "binary") {
        pA <- pA[, 2]; pB <- pB[, 2]
      }

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

      rel_changes <- abs(Ct_current - Ct_prev) / pmax(Ct_prev, 1e-10)
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

  # --- Build intervals ---
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

  if (object$outcome_type == "continuous") {
    if (!is.null(nuisance$sigma2_hat) && nk == n) {
      sigma2_at_new <- nuisance$sigma2_hat
    } else {
      warning("Prediction intervals at new points require sigma2 estimation at those points. ",
              "Using training-data sigma2 estimates.")
      sigma2_at_new <- nuisance$sigma2_hat
    }
    pi_var <- sigma2_at_new + mc_var + Ct_hat
    pi_se <- sqrt(pi_var)
    out$pi_lower <- f_hat - z_crit * pi_se
    out$pi_upper <- f_hat + z_crit * pi_se
    out$sigma2_hat <- sigma2_at_new
  }

  out
}
