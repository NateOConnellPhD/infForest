#' Estimate Nuisance Model for PASR
#'
#' Fits a cross-fitted nuisance model for synthetic outcome generation in PASR.
#' For continuous outcomes: location-scale model m(x) + sigma(x) * Z.
#' For binary outcomes: Bernoulli(p_hat(x)).
#'
#' @param object An \code{infForest} object.
#' @param R_cf Number of cross-fitting rounds for nuisance estimation. Default 5.
#' @param B_loc Number of trees for location (mean) forest. Default 1000.
#' @param B_scale Number of trees for scale (variance) forest. Default 1500.
#'
#' @return A list of class \code{infForest_nuisance} containing:
#' \describe{
#'   \item{outcome_type}{Character: "continuous" or "binary".}
#'   \item{m_hat}{Numeric vector of fitted conditional means (length n).}
#'   \item{sigma2_hat}{Numeric vector of fitted conditional variances (continuous only).}
#'   \item{p_hat}{Numeric vector of fitted probabilities (binary only).}
#' }
#'
#' @keywords internal
estimate_nuisance <- function(object, R_cf = 5L, B_loc = 1000L, B_scale = 1500L) {

  X <- object$X
  Y <- object$Y
  n <- nrow(X)
  p <- ncol(X)

  if (object$outcome_type == "binary") {
    # Binary: cross-fitted probability estimates
    p_sum <- numeric(n)
    p_cnt <- integer(n)

    for (r in seq_len(R_cf)) {
      idxA <- sample.int(n, n %/% 2)
      idxB <- setdiff(seq_len(n), idxA)

      # Fit probability forest on fold A, predict fold B
      datA <- X[idxA, , drop = FALSE]
      datA$y <- factor(Y[idxA], levels = c(0, 1))
      rfA <- inf.ranger::ranger(y ~ ., data = datA, probability = TRUE,
                                num.trees = B_loc, mtry = p,
                                min.node.size = 1L, sample.fraction = 1.0,
                                replace = FALSE, num.threads = 1L)
      predB <- predict(rfA, data = X[idxB, , drop = FALSE])$predictions[, 2]
      p_sum[idxB] <- p_sum[idxB] + predB
      p_cnt[idxB] <- p_cnt[idxB] + 1L

      # Fit on fold B, predict fold A
      datB <- X[idxB, , drop = FALSE]
      datB$y <- factor(Y[idxB], levels = c(0, 1))
      rfB <- inf.ranger::ranger(y ~ ., data = datB, probability = TRUE,
                                num.trees = B_loc, mtry = p,
                                min.node.size = 1L, sample.fraction = 1.0,
                                replace = FALSE, num.threads = 1L)
      predA <- predict(rfB, data = X[idxA, , drop = FALSE])$predictions[, 2]
      p_sum[idxA] <- p_sum[idxA] + predA
      p_cnt[idxA] <- p_cnt[idxA] + 1L
    }

    p_hat <- p_sum / pmax(p_cnt, 1L)
    p_hat <- pmin(pmax(p_hat, 1e-4), 1 - 1e-4)

    out <- list(outcome_type = "binary", p_hat = p_hat)

  } else {
    # Continuous: cross-fitted location-scale model
    # Step 1: cross-fitted mean estimates
    m1_sum <- numeric(n); m1_cnt <- integer(n)
    m2_sum <- numeric(n); m2_cnt <- integer(n)

    for (r in seq_len(R_cf)) {
      idxA <- sample.int(n, n %/% 2)
      idxB <- setdiff(seq_len(n), idxA)

      # First mean forest
      datA <- X[idxA, , drop = FALSE]; datA$y <- Y[idxA]
      rfA <- inf.ranger::ranger(y ~ ., data = datA, num.trees = B_loc,
                                mtry = p, min.node.size = 1L,
                                sample.fraction = 1.0, replace = FALSE,
                                num.threads = 1L)
      m1_sum[idxB] <- m1_sum[idxB] + predict(rfA, data = X[idxB, , drop = FALSE])$predictions
      m1_cnt[idxB] <- m1_cnt[idxB] + 1L

      datB <- X[idxB, , drop = FALSE]; datB$y <- Y[idxB]
      rfB <- inf.ranger::ranger(y ~ ., data = datB, num.trees = B_loc,
                                mtry = p, min.node.size = 1L,
                                sample.fraction = 1.0, replace = FALSE,
                                num.threads = 1L)
      m1_sum[idxA] <- m1_sum[idxA] + predict(rfB, data = X[idxA, , drop = FALSE])$predictions
      m1_cnt[idxA] <- m1_cnt[idxA] + 1L

      # Second independent mean forest (for cross-product residuals)
      idxC <- sample.int(n, n %/% 2)
      idxD <- setdiff(seq_len(n), idxC)

      datC <- X[idxC, , drop = FALSE]; datC$y <- Y[idxC]
      rfC <- inf.ranger::ranger(y ~ ., data = datC, num.trees = B_loc,
                                mtry = p, min.node.size = 1L,
                                sample.fraction = 1.0, replace = FALSE,
                                num.threads = 1L)
      m2_sum[idxD] <- m2_sum[idxD] + predict(rfC, data = X[idxD, , drop = FALSE])$predictions
      m2_cnt[idxD] <- m2_cnt[idxD] + 1L

      datD <- X[idxD, , drop = FALSE]; datD$y <- Y[idxD]
      rfD <- inf.ranger::ranger(y ~ ., data = datD, num.trees = B_loc,
                                mtry = p, min.node.size = 1L,
                                sample.fraction = 1.0, replace = FALSE,
                                num.threads = 1L)
      m2_sum[idxC] <- m2_sum[idxC] + predict(rfD, data = X[idxC, , drop = FALSE])$predictions
      m2_cnt[idxC] <- m2_cnt[idxC] + 1L
    }

    mhat1 <- m1_sum / pmax(m1_cnt, 1L)
    mhat2 <- m2_sum / pmax(m2_cnt, 1L)

    # Step 2: cross-product residuals -> variance estimate
    sprod <- (Y - mhat1) * (Y - mhat2)

    dat_scale <- X
    dat_scale$sprod <- sprod
    rf_scale <- inf.ranger::ranger(sprod ~ ., data = dat_scale,
                                   num.trees = B_scale, mtry = p,
                                   min.node.size = 10L,
                                   sample.fraction = 1.0, replace = FALSE,
                                   num.threads = 1L)
    sigma2_hat <- pmax(predict(rf_scale, data = X)$predictions, 1e-8)

    # m_hat is average of the two independent estimates
    m_hat <- (mhat1 + mhat2) / 2

    out <- list(outcome_type = "continuous", m_hat = m_hat, sigma2_hat = sigma2_hat,
                rf_scale = rf_scale)
  }

  class(out) <- "infForest_nuisance"
  out
}


#' Generate Synthetic Outcome Vector
#'
#' Draws one synthetic outcome vector from the fitted nuisance model.
#'
#' @param nuisance An \code{infForest_nuisance} object.
#' @param seed Random seed for reproducibility.
#'
#' @return Numeric vector of length n (synthetic outcomes).
#'
#' @keywords internal
generate_synthetic_Y <- function(nuisance, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- length(if (nuisance$outcome_type == "binary") nuisance$p_hat else nuisance$m_hat)

  if (nuisance$outcome_type == "binary") {
    rbinom(n, 1, nuisance$p_hat)
  } else {
    nuisance$m_hat + sqrt(nuisance$sigma2_hat) * rnorm(n)
  }
}
