#' Fit PASR variance model for a ranger object
#'
#' One-time cost: fits nuisance model and paired forests. The returned
#' \code{pasr_ranger} object can then predict at any new data via
#' \code{predict()}.
#'
#' @param object A fitted \code{ranger} object.
#' @param data Training data frame (must include response column).
#' @param x_conditional If TRUE (default), PASR conditional on training X.
#'   If FALSE, bootstrap-PASR for unconditional variance.
#' @param boot Number of bootstrap samples (only when x_conditional = FALSE).
#' @param R Number of PASR replicates per sample. Default 80.
#' @param B_mc Trees per PASR forest. Default 500.
#' @param alpha Significance level. Default 0.05.
#' @param verbose Print progress. Default FALSE.
#' @param ... Additional arguments.
#'
#' @return A \code{pasr_ranger} object. Use \code{predict()} for intervals.
#'
#' @export
pasr_predict <- function(object, data, x_conditional = TRUE,
                          R = 80L, B_mc = 500L, alpha = 0.05,
                          B_loc = 1000L, B_scale = 1500L, R_cf = 5L,
                          verbose = FALSE, ...) {
  if (!inherits(object, "ranger"))
    stop("pasr_predict() requires a fitted ranger object. For infForest, use predict().")

  resp_name <- setdiff(names(data), object$forest$independent.variable.names)
  if (length(resp_name) != 1) stop("Cannot identify response column in data.")
  Y_raw <- data[[resp_name]]
  vn <- object$forest$independent.variable.names
  X <- data[, vn, drop = FALSE]
  n <- nrow(X); p <- ncol(X)

  is_prob_forest <- !is.null(object$treetype) && object$treetype == "Probability estimation"
  if ((is.factor(Y_raw) && nlevels(Y_raw) == 2) || is_prob_forest) {
    outcome_type <- "binary"
    Y <- if (is.factor(Y_raw)) as.numeric(Y_raw) - 1 else as.numeric(Y_raw)
  } else {
    outcome_type <- "continuous"
    Y <- as.numeric(Y_raw)
  }

  has_future <- requireNamespace("future.apply", quietly = TRUE)
  use_parallel <- has_future && !inherits(future::plan(), "sequential")

  .strip_ranger <- function(rf) {
    out <- list(forest = rf$forest, treetype = rf$treetype, num.trees = rf$num.trees,
                importance.mode = rf$importance.mode,
                replace = rf$replace,
                dependent.variable.name = rf$dependent.variable.name)
    class(out) <- "ranger"
    out
  }

  # --- Nuisance fitting ---
  sigma2_model <- NULL; mhat <- NULL; sigma_hat_train <- NULL; pred_train <- NULL

  .fit_nuisance <- function(X_loc, Y_loc, n_loc, p_loc) {
    if (outcome_type == "continuous") {
      .fit_mean_pred <- function(idx_tr, idx_te, seed) {
        dat_tr <- X_loc[idx_tr, , drop = FALSE]; dat_tr$y <- Y_loc[idx_tr]
        rf <- inf.ranger::ranger(y ~ ., data = dat_tr, num.trees = B_loc,
                                  mtry = p_loc, min.node.size = 1,
                                  sample.fraction = 1.0, replace = FALSE,
                                  seed = seed, num.threads = 1L)
        as.numeric(predict(rf, data = X_loc[idx_te, , drop = FALSE])$predictions)
      }
      m1_sum <- numeric(n_loc); m1_cnt <- integer(n_loc)
      m2_sum <- numeric(n_loc); m2_cnt <- integer(n_loc)
      for (r in seq_len(R_cf)) {
        idxA <- sample.int(n_loc, n_loc %/% 2); idxB <- setdiff(seq_len(n_loc), idxA)
        m1_sum[idxB] <- m1_sum[idxB] + .fit_mean_pred(idxA, idxB, r*100L+1L)
        m1_sum[idxA] <- m1_sum[idxA] + .fit_mean_pred(idxB, idxA, r*100L+2L)
        m1_cnt[idxB] <- m1_cnt[idxB]+1L; m1_cnt[idxA] <- m1_cnt[idxA]+1L
        idxC <- sample.int(n_loc, n_loc %/% 2); idxD <- setdiff(seq_len(n_loc), idxC)
        m2_sum[idxD] <- m2_sum[idxD] + .fit_mean_pred(idxC, idxD, r*100L+3L)
        m2_sum[idxC] <- m2_sum[idxC] + .fit_mean_pred(idxD, idxC, r*100L+4L)
        m2_cnt[idxD] <- m2_cnt[idxD]+1L; m2_cnt[idxC] <- m2_cnt[idxC]+1L
      }
      mhat1 <- m1_sum / pmax(m1_cnt, 1L)
      mhat2 <- m2_sum / pmax(m2_cnt, 1L)
      sprod <- (Y_loc - mhat1) * (Y_loc - mhat2)

      dat_scale <- X_loc; dat_scale$sprod <- sprod
      rf_scale <- inf.ranger::ranger(sprod ~ ., data = dat_scale, num.trees = B_scale,
                                      mtry = p_loc, min.node.size = object$min.node.size,
                                      sample.fraction = 1.0, replace = TRUE,
                                      num.threads = 1L)
      mhat_loc <- (mhat1 + mhat2) / 2
      sigma_hat_loc <- sqrt(pmax(as.numeric(
        predict(rf_scale, data = X_loc)$predictions), 1e-8))

      list(sigma2_model = .strip_ranger(rf_scale),
           mhat = mhat_loc, sigma_hat_train = sigma_hat_loc)
    } else {
      pred <- predict(object, data = X_loc[, vn, drop = FALSE])$predictions
      if (is.matrix(pred)) pred <- pred[, 2]
      pred <- pmin(pmax(pred, 0.001), 0.999)
      list(pred_train = pred)
    }
  }

  # --- Fit paired forests for one sample, return stripped ---
  .make_pasr_fit_fn <- function(X_local, resp_name_local, vn_local,
                                 outcome_type_local, mhat_local,
                                 sigma_hat_train_local, pred_train_local,
                                 n_local, B_mc_local,
                                 mtry_local, min_node_local) {
    rf_args_base <- list(
      formula = as.formula(paste(resp_name_local, "~ .")),
      num.trees = B_mc_local, mtry = mtry_local,
      min.node.size = min_node_local,
      sample.fraction = 1.0, replace = FALSE,
      num.threads = 1L, write.forest = TRUE
    )
    if (outcome_type_local == "binary") rf_args_base$probability <- TRUE

    function(r) {
      set.seed(r * 7919L)
      if (outcome_type_local == "continuous") {
        Y_syn <- mhat_local + sigma_hat_train_local * rnorm(n_local)
      } else {
        Y_syn <- factor(rbinom(n_local, 1, pred_train_local), levels = c(0, 1))
      }
      dat_syn <- X_local; dat_syn[[resp_name_local]] <- Y_syn
      rf_args <- rf_args_base; rf_args$data <- dat_syn
      rf_args$seed <- r * 100L + 1L
      rfA <- do.call(inf.ranger::ranger, rf_args)
      rf_args$seed <- r * 100L + 2L
      rfB <- do.call(inf.ranger::ranger, rf_args)
      .s <- function(rf) {
        out <- list(forest = rf$forest, treetype = rf$treetype, num.trees = rf$num.trees,
                    importance.mode = rf$importance.mode,
                    replace = rf$replace,
                    dependent.variable.name = rf$dependent.variable.name)
        class(out) <- "ranger"
        out
      }
      list(rfA = .s(rfA), rfB = .s(rfB))
    }
  }

  if (x_conditional) {
    # --- Conditional PASR ---
    if (verbose) cat("Fitting nuisance model...\n")
    nuis <- .fit_nuisance(X, Y, n, p)

    if (verbose) cat(sprintf("Fitting %d paired forests...\n", R))
    .fit_one <- .make_pasr_fit_fn(X, resp_name, vn, outcome_type,
                                    nuis$mhat, nuis$sigma_hat_train,
                                    nuis$pred_train, n, B_mc,
                                    object$mtry, object$min.node.size)

    if (use_parallel) {
      paired_forests <- future.apply::future_lapply(
        seq_len(R), .fit_one, future.seed = TRUE,
        future.packages = "inf.ranger")
    } else {
      paired_forests <- lapply(seq_len(R), function(r) {
        if (verbose && r %% 20 == 0) cat(sprintf("  Replicate %d/%d\n", r, R))
        .fit_one(r)
      })
    }

    out <- list(
      deployed = .strip_ranger(object),
      paired_forests = paired_forests,
      sigma2_model = nuis$sigma2_model,
      outcome_type = outcome_type,
      vn = vn,
      alpha = alpha,
      R = R,
      x_conditional = TRUE
    )
    class(out) <- "pasr_ranger"

  } else {
    # --- Unconditional: PASR (terms I+II) + design-point variance (term III) ---
    # Term III computed analytically from forest weights — no bootstrap needed

    if (verbose) cat("Fitting nuisance model...\n")
    nuis <- .fit_nuisance(X, Y, n, p)

    if (verbose) cat(sprintf("Fitting %d paired forests...\n", R))
    .fit_one_cond <- .make_pasr_fit_fn(X, resp_name, vn, outcome_type,
                                        nuis$mhat, nuis$sigma_hat_train,
                                        nuis$pred_train, n, B_mc,
                                        object$mtry, object$min.node.size)

    if (use_parallel) {
      paired_forests <- future.apply::future_lapply(
        seq_len(R), .fit_one_cond, future.seed = TRUE,
        future.packages = "inf.ranger")
    } else {
      paired_forests <- lapply(seq_len(R), function(r) {
        if (verbose && r %% 20 == 0) cat(sprintf("  Replicate %d/%d\n", r, R))
        .fit_one_cond(r)
      })
    }

    # Compute terminal node IDs for training data (for V_X at predict time)
    if (verbose) cat("Computing training terminal nodes...\n")
    if (is.null(object$inbag.counts))
      stop("Unconditional PASR requires keep.inbag = TRUE in the ranger call.")
    train_nodes <- predict(object, data = X, type = "terminalNodes")$predictions
    storage.mode(train_nodes) <- "integer"

    # Inbag matrix: n x B, >0 if observation was in-bag for that tree
    inbag_mat <- do.call(cbind, object$inbag.counts)
    storage.mode(inbag_mat) <- "integer"

    # Deployed forest predictions at training points (for V_X weighted variance)
    f_hat_train <- as.numeric(predict(object, data = X)$predictions)
    if (outcome_type == "binary") {
      pred_tmp <- predict(object, data = X)$predictions
      f_hat_train <- if (is.matrix(pred_tmp)) pred_tmp[, 2] else pred_tmp
    }

    out <- list(
      deployed = .strip_ranger(object),
      paired_forests = paired_forests,
      sigma2_model = nuis$sigma2_model,
      train_leaf_ids = train_nodes,
      inbag = inbag_mat,
      f_hat_train = f_hat_train,
      outcome_type = outcome_type,
      vn = vn,
      alpha = alpha,
      R = R,
      x_conditional = FALSE
    )
    class(out) <- "pasr_ranger"
  }

  if (verbose) cat("PASR fitting complete.\n")
  out
}


#' @export
print.pasr_ranger <- function(x, ...) {
  cat("PASR Ranger Object\n")
  cat("  Outcome type:   ", x$outcome_type, "\n")
  cat("  X-conditional:  ", x$x_conditional, "\n")
  cat("  PASR replicates:", x$R, "\n")
  if (!x$x_conditional) {
    cat("  Term III:        design-point variance (analytic)\n")
  }
  invisible(x)
}


#' @export
predict.pasr_ranger <- function(object, newdata = NULL, alpha = NULL,
                                 unconditional = NULL, ...) {
  if (is.null(alpha)) alpha <- object$alpha
  # Default: unconditional if object was fit that way
  if (is.null(unconditional)) unconditional <- !object$x_conditional
  # Can only do unconditional if object has the required data
  if (unconditional && object$x_conditional)
    stop("Cannot compute unconditional intervals from a conditional fit. Refit with x_conditional = FALSE.")
  z_crit <- qnorm(1 - alpha / 2)
  vn <- object$vn
  outcome_type <- object$outcome_type

  if (is.null(newdata)) stop("newdata is required for predict.pasr_ranger.")
  nd <- newdata[, vn, drop = FALSE]
  nk <- nrow(nd)
  nt <- .get_n_threads()
  is_binary <- (outcome_type == "binary")

  # Deployed forest predictions
  pred_all <- predict(object$deployed, data = nd, predict.all = TRUE)$predictions
  if (outcome_type == "binary") pred_all <- pred_all[, 2, ]
  f_hat <- rowMeans(pred_all)
  if (outcome_type == "binary") f_hat <- pmin(pmax(f_hat, 1e-4), 1 - 1e-4)
  s2_trees <- rowSums((pred_all - f_hat)^2) / (ncol(pred_all) - 1)
  mc_var <- s2_trees / ncol(pred_all)

  if (!unconditional) {
    # --- Conditional predict from pre-extracted cache ---
    R <- object$R

    .rpred <- function(rf, nd_loc) {
      p <- predict(rf, data = nd_loc)$predictions
      if (outcome_type == "binary" && is.matrix(p)) p <- p[, 2]
      as.numeric(p)
    }
    psi_A_mat <- matrix(NA_real_, nk, R)
    psi_B_mat <- matrix(NA_real_, nk, R)
    for (r in seq_len(R)) {
      psi_A_mat[, r] <- .rpred(object$paired_forests[[r]]$rfA, nd)
      psi_B_mat[, r] <- .rpred(object$paired_forests[[r]]$rfB, nd)
    }

    Ct_hat <- numeric(nk)
    for (k in seq_len(nk)) Ct_hat[k] <- max(cov(psi_A_mat[k, ], psi_B_mat[k, ]), 0)

    ci_var <- mc_var + Ct_hat; ci_se <- sqrt(ci_var)
    out <- data.frame(
      f_hat = f_hat, se = ci_se,
      ci_lower = f_hat - z_crit * ci_se, ci_upper = f_hat + z_crit * ci_se,
      mc_var = mc_var, Ct_hat = Ct_hat, R_used = R
    )

    if (outcome_type == "continuous" && !is.null(object$sigma2_model)) {
      sigma2_hat <- pmax(as.numeric(
        predict(object$sigma2_model, data = nd)$predictions), 1e-8)
      pi_var <- sigma2_hat + mc_var + Ct_hat; pi_se <- sqrt(pi_var)
      out$pi_lower <- f_hat - z_crit * pi_se
      out$pi_upper <- f_hat + z_crit * pi_se
      out$sigma2_hat <- sigma2_hat
    }

  } else {
    # --- Unconditional predict ---
    R <- object$R

    # Term I+II: paired forests
    .rpred <- function(rf, nd_loc) {
      p <- predict(rf, data = nd_loc)$predictions
      if (outcome_type == "binary" && is.matrix(p)) p <- p[, 2]
      as.numeric(p)
    }
    psi_A_mat <- matrix(NA_real_, nk, R)
    psi_B_mat <- matrix(NA_real_, nk, R)
    for (r in seq_len(R)) {
      psi_A_mat[, r] <- .rpred(object$paired_forests[[r]]$rfA, nd)
      psi_B_mat[, r] <- .rpred(object$paired_forests[[r]]$rfB, nd)
    }
    Ct_hat <- numeric(nk)
    for (k in seq_len(nk)) Ct_hat[k] <- max(cov(psi_A_mat[k, ], psi_B_mat[k, ]), 0)
    cond_var <- mc_var + Ct_hat

    # Term III: design-point variance from forest weights
    is_binary <- (outcome_type == "binary")
    nt <- .get_n_threads()
    vx_result <- compute_design_point_variance_cpp(
      object$deployed$forest,
      as.matrix(nd),
      object$train_leaf_ids,
      object$inbag,
      object$f_hat_train,
      is_binary,
      nt
    )
    var_x <- vx_result$V_X

    ci_var_uncond <- cond_var + var_x
    ci_se_uncond <- sqrt(ci_var_uncond)

    out <- data.frame(
      f_hat = f_hat, se = ci_se_uncond,
      ci_lower = f_hat - z_crit * ci_se_uncond,
      ci_upper = f_hat + z_crit * ci_se_uncond,
      var_conditional = cond_var,
      var_x = var_x,
      n_eff = vx_result$n_eff
    )

    if (outcome_type == "continuous" && !is.null(object$sigma2_model)) {
      sigma2_hat <- pmax(as.numeric(
        predict(object$sigma2_model, data = nd)$predictions), 1e-8)
      pi_var <- sigma2_hat + ci_var_uncond
      pi_se <- sqrt(pi_var)
      out$pi_lower <- f_hat - z_crit * pi_se
      out$pi_upper <- f_hat + z_crit * pi_se
      out$sigma2_hat <- sigma2_hat
    }
  }

  out
}


#' @rdname pasr_predict
#' @export
predict.infForest <- function(object, newdata = NULL, pred_type = NULL,
                               alpha = 0.05, R = 50L,
                               verbose = FALSE, propensities = NULL, ...) {
  check_infForest(object)
  pasr <- object$pasr

  if (is.null(newdata)) newdata <- object$X
  if (!is.data.frame(newdata)) newdata <- as.data.frame(newdata)

  n <- nrow(object$X)
  Y <- object$Y
  z_crit <- qnorm(1 - alpha / 2)

  model_vars <- colnames(object$X)
  specified_vars <- intersect(colnames(newdata), model_vars)
  margin_vars <- setdiff(model_vars, specified_vars)
  is_complete <- length(margin_vars) == 0
  is_marginal <- !is_complete

  if (length(specified_vars) == 0)
    stop("newdata must contain at least one variable from the fitted model.")

  if (is.null(pred_type))
    pred_type <- if (is_complete) "pointwise" else "marginal"
  pred_type <- match.arg(pred_type, c("pointwise", "marginal"))

  if (pred_type == "pointwise" && is_marginal)
    stop("pred_type = 'pointwise' requires all model variables in newdata.")

  n_queries <- nrow(newdata)
  Y_raw <- object$Y
  is_binary <- is.factor(Y_raw) && nlevels(Y_raw) == 2

  # Point estimates from observed data (pass propensities if available)
  mu_obs <- .infForest_predict_once(object, newdata, specified_vars,
                                     margin_vars, is_complete, n, Y,
                                     propensities = propensities)

  # Sigma2 for pointwise PI
  sigma2_hat <- NULL
  if (pred_type == "pointwise" && !is_binary)
    sigma2_hat <- .estimate_sigma2(object, newdata)

  # PASR variance
  if (!is.null(pasr)) {
    if (!inherits(pasr, "infForest_pasr"))
      stop("pasr must be an infForest_pasr object from pasr().")
    if (verbose) cat("Using pre-fit PASR object (R =", pasr$R, "replicates)\n")

    # Fit propensities for marginalized case (skip if pre-computed)
    if (is_marginal && is.null(propensities)) {
      propensities <- list()
      for (vname in specified_vars) {
        vtype <- detect_var_type(object$X[[vname]])
        is_bin <- (vtype == "binary")
        propensities[[vname]] <- list(
          ghat = .fit_propensity(object$X, vname, is_binary = is_bin,
                                  n_trees = 2000L)$ghat,
          is_binary = is_bin,
          x_var = object$X[[vname]]
        )
      }
    }

    # Precompute constants
    precomputed <- .precompute_predict_constants(object, newdata,
                                                  specified_vars, propensities)

    nt <- .get_n_threads()

    if (is_marginal) {
      # Build Y_syn list for C++
      Y_syn_list <- vector("list", pasr$R)
      for (r in seq_len(pasr$R)) Y_syn_list[[r]] <- pasr$Y_syn[, r]

      # Build omega matrix: n x n_queries
      omega_mat <- do.call(cbind, precomputed$omega_list)

      # Single C++ call: all replicates, all queries
      mu_matrix <- pasr_extract_all_marginal_cpp(
        pasr$caches, Y_syn_list, precomputed$X_cf_list, omega_mat,
        n_threads = nt)
    } else {
      # Complete data: use R loop with cached predict (fast for small n_queries)
      mu_matrix <- matrix(NA_real_, nrow = pasr$R, ncol = n_queries)
      for (r in seq_len(pasr$R)) {
        mu_matrix[r, ] <- .pasr_extract_predict(pasr, r, object, newdata,
                                                  specified_vars, propensities,
                                                  precomputed)
      }
    }
    se <- apply(mu_matrix, 2, sd)

  } else {
    # No PASR object — compute on the fly (slow)
    if (verbose) {
      if (is_marginal) {
        cat("Marginalized prediction over:", paste(margin_vars, collapse = ", "), "\n")
      } else {
        cat("Pointwise prediction (all variables specified)\n")
      }
      cat("Note: pass a pasr() object for faster computation.\n")
      cat("Computing PASR standard errors (R =", R, "replicates)...\n")
    }

    fhat_full <- .get_full_forest_predictions(object)
    if (is_binary) {
      p_hat <- pmin(pmax(fhat_full, 0.001), 0.999)
    } else {
      sigma_hat_train <- sqrt(pmax(mean((as.numeric(Y) - fhat_full)^2, na.rm = TRUE), 1e-8))
    }

    mu_matrix <- matrix(NA_real_, nrow = R, ncol = n_queries)
    for (r in seq_len(R)) {
      if (verbose && r %% 10 == 0) cat("  PASR replicate", r, "of", R, "\n")
      set.seed(r * 7919L)
      if (is_binary) {
        Y_syn <- factor(rbinom(n, 1, p_hat), levels = levels(Y_raw))
      } else {
        Y_syn <- fhat_full + rnorm(n, 0, sigma_hat_train)
      }
      mu_matrix[r, ] <- .infForest_predict_once(object, newdata, specified_vars,
                                                  margin_vars, is_complete, n, Y_syn)
    }
    se <- apply(mu_matrix, 2, sd)
  }

  ci_lower <- mu_obs - z_crit * se
  ci_upper <- mu_obs + z_crit * se

  out <- newdata[, specified_vars, drop = FALSE]
  out$estimate <- mu_obs
  out$se <- se
  out$ci_lower <- ci_lower
  out$ci_upper <- ci_upper

  if (pred_type == "pointwise" && !is_binary && !is.null(sigma2_hat)) {
    pi_se <- sqrt(se^2 + sigma2_hat)
    out$pi_lower <- mu_obs - z_crit * pi_se
    out$pi_upper <- mu_obs + z_crit * pi_se
    out$sigma2_hat <- sigma2_hat
  }

  if (is_marginal)
    out$marginalized_over <- paste(margin_vars, collapse = ", ")

  out$pred_type <- pred_type
  class(out) <- c("infForest_prediction", "data.frame")
  out
}


# ============================================================
# Internal: call ranger's own predict method (avoids recursion)
# ============================================================
#' @keywords internal
.ranger_predict <- function(object, ...) {
  # Get ranger's predict method from its namespace, bypassing ours
  ranger_ns <- asNamespace("inf.ranger")
  ranger_pred <- get("predict.ranger", envir = ranger_ns)
  ranger_pred(object, ...)
}


# ============================================================
# Internal: ranger PASR prediction (full decomposition)
# ============================================================
#' @keywords internal
.predict_ranger_pasr <- function(object, data, newdata = NULL,
                                  R_min = 20L, R_max = 200L, batch_size = 10L,
                                  tol = 0.05, n_stable = 2L,
                                  B_mc = 500L, alpha = 0.05,
                                  B_loc = 1000L, B_scale = 1500L, R_cf = 5L,
                                  verbose = FALSE, ...) {

  z_crit <- qnorm(1 - alpha / 2)

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
  n <- nrow(X); p <- ncol(X)
  vn <- object$forest$independent.variable.names

  if (is.null(newdata)) newdata <- X
  nk <- nrow(newdata)

  # Deployed forest predictions and MC variance
  pred_all <- .ranger_predict(object, data = newdata[, vn, drop = FALSE],
                               predict.all = TRUE)$predictions
  if (outcome_type == "binary") pred_all <- pred_all[, 2, ]

  f_hat <- rowMeans(pred_all)
  if (outcome_type == "binary") f_hat <- pmin(pmax(f_hat, 1e-4), 1 - 1e-4)
  s2_trees <- rowSums((pred_all - f_hat)^2) / (ncol(pred_all) - 1)
  mc_var <- s2_trees / ncol(pred_all)

  # Nuisance: continuous only
  sigma2_hat <- NULL; mhat <- NULL; sigma_hat_train <- NULL

  if (outcome_type == "continuous") {
    .fit_mean_pred <- function(idx_tr, idx_te, seed) {
      dat_tr <- X[idx_tr, , drop = FALSE]; dat_tr$y <- Y[idx_tr]
      rf <- inf.ranger::ranger(y ~ ., data = dat_tr, num.trees = B_loc,
                                mtry = p, min.node.size = 1,
                                sample.fraction = 1.0, replace = FALSE,
                                seed = seed, num.threads = 1L)
      as.numeric(.ranger_predict(rf, data = X[idx_te, , drop = FALSE])$predictions)
    }

    m1_sum <- numeric(n); m1_cnt <- integer(n)
    m2_sum <- numeric(n); m2_cnt <- integer(n)
    for (r in seq_len(R_cf)) {
      idxA <- sample.int(n, n %/% 2); idxB <- setdiff(seq_len(n), idxA)
      m1_sum[idxB] <- m1_sum[idxB] + .fit_mean_pred(idxA, idxB, r*100L+1L)
      m1_sum[idxA] <- m1_sum[idxA] + .fit_mean_pred(idxB, idxA, r*100L+2L)
      m1_cnt[idxB] <- m1_cnt[idxB]+1L; m1_cnt[idxA] <- m1_cnt[idxA]+1L
      idxC <- sample.int(n, n %/% 2); idxD <- setdiff(seq_len(n), idxC)
      m2_sum[idxD] <- m2_sum[idxD] + .fit_mean_pred(idxC, idxD, r*100L+3L)
      m2_sum[idxC] <- m2_sum[idxC] + .fit_mean_pred(idxD, idxC, r*100L+4L)
      m2_cnt[idxD] <- m2_cnt[idxD]+1L; m2_cnt[idxC] <- m2_cnt[idxC]+1L
    }
    mhat1 <- m1_sum / pmax(m1_cnt, 1L)
    mhat2 <- m2_sum / pmax(m2_cnt, 1L)
    sprod <- (Y - mhat1) * (Y - mhat2)

    dat_scale <- X; dat_scale$sprod <- sprod
    rf_scale <- inf.ranger::ranger(sprod ~ ., data = dat_scale, num.trees = B_scale,
                                    mtry = p, min.node.size = object$min.node.size,
                                    sample.fraction = 1.0, replace = FALSE,
                                    num.threads = 1L)
    sigma2_hat <- pmax(as.numeric(.ranger_predict(rf_scale,
                        data = newdata[, vn, drop = FALSE])$predictions), 1e-8)
    mhat <- (mhat1 + mhat2) / 2
    sigma_hat_train <- sqrt(pmax(as.numeric(
      .ranger_predict(rf_scale, data = X)$predictions), 1e-8))
    if (verbose) cat("  Nuisance: cross-fitted residual product complete\n")
  }

  # Binary: fitted probs for synthetic Y
  if (outcome_type == "binary") {
    pred_train <- .ranger_predict(object, data = X[, vn, drop = FALSE])$predictions
    if (is.matrix(pred_train)) pred_train <- pred_train[, 2]
    pred_train <- pmin(pmax(pred_train, 0.001), 0.999)
  }

  # PASR: paired forests
  has_future <- requireNamespace("future.apply", quietly = TRUE)
  use_parallel <- has_future && !inherits(future::plan(), "sequential")

  # Pre-allocate all replicates at R_max, trim later
  psi_A_mat <- matrix(NA, nk, R_max); psi_B_mat <- matrix(NA, nk, R_max)
  R_current <- 0L; Ct_prev <- rep(Inf, nk)
  stable_count <- 0L; converged <- FALSE

  rf_args_base <- list(
    formula = as.formula(paste(resp_name, "~ .")),
    num.trees = B_mc, mtry = object$mtry,
    min.node.size = object$min.node.size,
    sample.fraction = 1.0, replace = FALSE,
    num.threads = 1L, write.forest = TRUE
  )
  if (outcome_type == "binary") rf_args_base$probability <- TRUE

  # Factory for isolated closure (avoids shipping large objects to workers)
  .make_pasr_ranger_fn <- function(X_local, resp_name_local, vn_local,
                                     newdata_local, rf_args_base_local,
                                     outcome_type_local, mhat_local,
                                     sigma_hat_train_local, pred_train_local, n_local) {
    function(r) {
      set.seed(r * 7919L)
      if (outcome_type_local == "continuous") {
        Y_syn <- mhat_local + sigma_hat_train_local * rnorm(n_local)
      } else {
        Y_syn <- factor(rbinom(n_local, 1, pred_train_local), levels = c(0, 1))
      }
      dat_syn <- X_local; dat_syn[[resp_name_local]] <- Y_syn
      rf_args <- rf_args_base_local; rf_args$data <- dat_syn
      rf_args$seed <- r * 100L + 1L
      rfA <- do.call(inf.ranger::ranger, rf_args)
      rf_args$seed <- r * 100L + 2L
      rfB <- do.call(inf.ranger::ranger, rf_args)
      pA <- predict(rfA, data = newdata_local[, vn_local, drop = FALSE])$predictions
      pB <- predict(rfB, data = newdata_local[, vn_local, drop = FALSE])$predictions
      if (outcome_type_local == "binary") { pA <- pA[, 2]; pB <- pB[, 2] }
      list(pA = as.numeric(pA), pB = as.numeric(pB))
    }
  }

  .fit_one_ranger_pasr <- .make_pasr_ranger_fn(
    X, resp_name, vn, newdata, rf_args_base, outcome_type,
    mhat, sigma_hat_train,
    if (outcome_type == "binary") pred_train else NULL, n)

  while (R_current < R_max) {
    r_start <- R_current + 1L
    r_end <- min(R_current + batch_size, R_max)
    r_seq <- seq(r_start, r_end)

    if (use_parallel) {
      batch_res <- future.apply::future_lapply(r_seq, .fit_one_ranger_pasr,
                                                future.seed = TRUE,
                                                future.packages = "inf.ranger")
    } else {
      batch_res <- lapply(r_seq, .fit_one_ranger_pasr)
    }

    for (idx in seq_along(r_seq)) {
      r <- r_seq[idx]
      psi_A_mat[, r] <- batch_res[[idx]]$pA
      psi_B_mat[, r] <- batch_res[[idx]]$pB
    }
    R_current <- r_end

    if (R_current >= R_min) {
      Ct_current <- numeric(nk)
      cols <- seq_len(R_current)
      for (k in seq_len(nk)) Ct_current[k] <- max(cov(psi_A_mat[k, cols], psi_B_mat[k, cols]), 0)
      rel_changes <- ifelse(is.finite(Ct_prev),
                            abs(Ct_current - Ct_prev) / pmax(Ct_prev, 1e-10), Inf)
      med_rel_change <- median(rel_changes)
      if (verbose)
        cat(sprintf("  PASR R=%d: median_Ct=%.6f  med_rel_change=%.4f  stable=%d/%d\n",
                    R_current, median(Ct_current), med_rel_change, stable_count, n_stable))
      if (med_rel_change < tol) stable_count <- stable_count + 1L else stable_count <- 0L
      if (stable_count >= n_stable) { converged <- TRUE; break }
      Ct_prev <- Ct_current
    }
  }

  Ct_hat <- numeric(nk)
  cols <- seq_len(R_current)
  for (k in seq_len(nk)) Ct_hat[k] <- max(cov(psi_A_mat[k, cols], psi_B_mat[k, cols]), 0)

  if (verbose) {
    if (converged) cat(sprintf("  PASR converged at R=%d (median Ct=%.6f)\n", R_current, median(Ct_hat)))
    else cat(sprintf("  PASR reached R_max=%d without convergence (median Ct=%.6f)\n", R_max, median(Ct_hat)))
  }

  ci_var <- mc_var + Ct_hat; ci_se <- sqrt(ci_var)
  out <- data.frame(
    f_hat = f_hat, se = ci_se,
    ci_lower = f_hat - z_crit * ci_se, ci_upper = f_hat + z_crit * ci_se,
    mc_var = mc_var, Ct_hat = Ct_hat, R_used = R_current, converged = converged
  )
  if (outcome_type == "continuous" && !is.null(sigma2_hat)) {
    pi_var <- sigma2_hat + mc_var + Ct_hat; pi_se <- sqrt(pi_var)
    out$pi_lower <- f_hat - z_crit * pi_se
    out$pi_upper <- f_hat + z_crit * pi_se
    out$sigma2_hat <- sigma2_hat
  }
  out
}


# ============================================================
# Unconditional PASR: bootstrap X, run PASR within each
# Captures Var(f_hat) = E_X[Var(f_hat|X)] + Var_X[E(f_hat|X)]
# ============================================================
#' @keywords internal
.predict_ranger_pasr_unconditional <- function(object, data, newdata = NULL,
                                                R_min = 20L, R_max = 100L,
                                                batch_size = 10L,
                                                tol = 0.05, n_stable = 2L,
                                                B_mc = 500L, alpha = 0.05,
                                                B_loc = 1000L, B_scale = 1500L,
                                                R_cf = 5L, boot = 20L,
                                                verbose = FALSE, ...) {

  z_crit <- qnorm(1 - alpha / 2)
  resp_name <- setdiff(names(data), object$forest$independent.variable.names)
  if (length(resp_name) != 1) stop("Cannot identify response column in data.")
  vn <- object$forest$independent.variable.names
  n <- nrow(data)

  if (is.null(newdata)) newdata <- data[, vn, drop = FALSE]
  nk <- nrow(newdata)

  # Deployed forest prediction at query points (from original fit)
  pred_all <- .ranger_predict(object, data = newdata[, vn, drop = FALSE],
                               predict.all = TRUE)$predictions
  Y_raw <- data[[resp_name]]
  is_prob_forest <- !is.null(object$treetype) && object$treetype == "Probability estimation"
  outcome_type <- if ((is.factor(Y_raw) && nlevels(Y_raw) == 2) || is_prob_forest) "binary" else "continuous"
  if (outcome_type == "binary") pred_all <- pred_all[, 2, ]
  f_hat <- rowMeans(pred_all)
  if (outcome_type == "binary") f_hat <- pmin(pmax(f_hat, 1e-4), 1 - 1e-4)

  has_future <- requireNamespace("future.apply", quietly = TRUE)
  use_parallel <- has_future && !inherits(future::plan(), "sequential")

  if (verbose) cat(sprintf("Unconditional PASR: %d bootstrap samples, R_max=%d each\n", boot, R_max))

  # Factory for bootstrap-PASR replicate (isolated closure)
  .make_boot_fn <- function(data_local, object_local, newdata_local, vn_local,
                             resp_name_local, R_max_local, R_min_local,
                             batch_size_local, tol_local, n_stable_local,
                             B_mc_local, B_loc_local, B_scale_local,
                             R_cf_local, alpha_local, n_local) {
    function(b) {
      set.seed(b * 31337L)
      boot_idx <- sample.int(n_local, n_local, replace = TRUE)
      data_boot <- data_local[boot_idx, , drop = FALSE]

      Y_raw_b <- data_boot[[resp_name_local]]
      is_prob <- is.factor(Y_raw_b) && nlevels(Y_raw_b) == 2
      rf_args <- list(
        formula = as.formula(paste(resp_name_local, "~ .")),
        data = data_boot,
        num.trees = object_local$num.trees,
        mtry = object_local$mtry,
        min.node.size = object_local$min.node.size,
        num.threads = 1L
      )
      if (is_prob) rf_args$probability <- TRUE
      rf_boot <- do.call(inf.ranger::ranger, rf_args)

      result <- .predict_ranger_pasr(
        rf_boot, data = data_boot, newdata = newdata_local,
        R_min = R_min_local, R_max = R_max_local,
        batch_size = batch_size_local, tol = tol_local,
        n_stable = n_stable_local, B_mc = B_mc_local,
        alpha = alpha_local, B_loc = B_loc_local,
        B_scale = B_scale_local, R_cf = R_cf_local,
        verbose = FALSE)

      list(
        f_hat_boot = result$f_hat,
        ci_var_boot = result$mc_var + result$Ct_hat,
        sigma2_boot = if ("sigma2_hat" %in% names(result)) result$sigma2_hat else NULL
      )
    }
  }

  .run_one_boot <- .make_boot_fn(
    data, object, newdata, vn, resp_name,
    R_max, R_min, batch_size, tol, n_stable,
    B_mc, B_loc, B_scale, R_cf, alpha, n)

  if (use_parallel) {
    boot_results <- future.apply::future_lapply(
      seq_len(boot), .run_one_boot,
      future.seed = TRUE,
      future.packages = "inf.ranger")
  } else {
    boot_results <- lapply(seq_len(boot), function(b) {
      if (verbose && b %% 5 == 0) cat(sprintf("  Bootstrap %d/%d\n", b, boot))
      .run_one_boot(b)
    })
  }

  fhat_mat <- matrix(NA_real_, nk, boot)
  civar_mat <- matrix(NA_real_, nk, boot)
  sigma2_mat <- if (outcome_type == "continuous") matrix(NA_real_, nk, boot) else NULL

  for (b in seq_len(boot)) {
    fhat_mat[, b] <- boot_results[[b]]$f_hat_boot
    civar_mat[, b] <- boot_results[[b]]$ci_var_boot
    if (!is.null(sigma2_mat)) sigma2_mat[, b] <- boot_results[[b]]$sigma2_boot
  }

  # Law of total variance:
  # E_X*[Var(f|X*)]: average conditional PASR variance across bootstraps
  mean_cond_var <- rowMeans(civar_mat)
  # Var_X*[f(x)]: variance of point estimates across bootstraps
  var_fhat <- apply(fhat_mat, 1, var)
  # Total unconditional CI variance
  ci_var_uncond <- mean_cond_var + var_fhat
  ci_se_uncond <- sqrt(ci_var_uncond)

  out <- data.frame(
    f_hat = f_hat,
    se = ci_se_uncond,
    ci_lower = f_hat - z_crit * ci_se_uncond,
    ci_upper = f_hat + z_crit * ci_se_uncond,
    var_conditional = mean_cond_var,
    var_x = var_fhat,
    boot_samples = boot
  )

  if (outcome_type == "continuous" && !is.null(sigma2_mat)) {
    mean_sigma2 <- rowMeans(sigma2_mat)
    pi_var <- mean_sigma2 + ci_var_uncond
    pi_se <- sqrt(pi_var)
    out$pi_lower <- f_hat - z_crit * pi_se
    out$pi_upper <- f_hat + z_crit * pi_se
    out$sigma2_hat <- mean_sigma2
  }

  if (verbose) {
    cat(sprintf("  Unconditional PASR complete: %d bootstraps\n", boot))
    cat(sprintf("  Median conditional var:  %.6f\n", median(mean_cond_var)))
    cat(sprintf("  Median X-variance:       %.6f\n", median(var_fhat)))
    cat(sprintf("  Median total var:        %.6f\n", median(ci_var_uncond)))
  }

  out
}


# ============================================================
# Internal: infForest predict once
# ============================================================
#' @keywords internal
.infForest_predict_once <- function(object, newdata, specified_vars,
                                     margin_vars, is_complete, n, Y,
                                     propensities = NULL) {
  n_queries <- nrow(newdata)
  if (is_complete) return(.predict_complete(object, newdata, n, Y))

  # Fit propensity for each specified variable once (skip if provided)
  if (is.null(propensities)) {
    propensities <- list()
    for (vname in specified_vars) {
      vtype <- detect_var_type(object$X[[vname]])
      is_bin <- (vtype == "binary")
      propensities[[vname]] <- list(
        ghat = .fit_propensity(object$X, vname, is_binary = is_bin,
                                n_trees = 2000L)$ghat,
        is_binary = is_bin,
        x_var = object$X[[vname]]
      )
    }
  }

  # Use forest caches for speed
  caches <- object$forest_caches
  use_cached <- !is.null(caches)

  # Precompute col_idx, sigma2, omega
  col_idxs <- list()
  for (vname in specified_vars)
    col_idxs[[vname]] <- get_ranger_col_idx(object$forests[[1]]$rfA, vname)

  sigma2_map <- list()
  for (vname in specified_vars) {
    prop <- propensities[[vname]]
    if (!prop$is_binary) {
      ej <- as.numeric(prop$x_var) - prop$ghat
      sigma2_map[[vname]] <- mean(ej^2)
    }
  }

  mu <- numeric(n_queries)
  for (q in seq_len(n_queries)) {
    query_vals <- newdata[q, specified_vars, drop = FALSE]

    # Build X_cf and omega once per query
    X_cf <- object$X_ord
    for (vname in specified_vars) {
      val <- query_vals[[vname]]
      if (is.factor(object$X[[vname]]))
        val <- as.numeric(factor(val, levels = levels(object$X[[vname]])))
      X_cf[, col_idxs[[vname]] + 1] <- val
    }

    omega <- numeric(n)
    for (vname in specified_vars) {
      prop <- propensities[[vname]]
      if (prop$is_binary) {
        x_j <- as.numeric(prop$x_var)
        gc <- pmax(0.025, pmin(0.975, prop$ghat))
        qv <- as.numeric(query_vals[[vname]])
        if (qv == 1) omega <- omega + x_j / gc
        else omega <- omega + (1 - x_j) / (1 - gc)
      } else {
        ej <- as.numeric(prop$x_var) - prop$ghat
        omega <- omega + ej / sigma2_map[[vname]]
      }
    }

    phi_sum <- 0; phi_cnt <- 0L

    if (use_cached) {
      # Cached path: use forest_caches
      cache_names <- names(caches)
      for (cn in cache_names) {
        cache <- caches[[cn]]
        hon <- as.integer(cache$honest_idx)

        preds_cf <- honest_predict_cached_cpp(cache, X_cf)
        fhat_obs_vec <- cache$fhat_obs

        for (j in seq_along(hon)) {
          k <- hon[j]
          fhat_cf <- preds_cf[k]
          fhat_obs_j <- fhat_obs_vec[j]
          if (is.na(fhat_cf) || is.na(fhat_obs_j)) next
          R_k <- as.numeric(Y[k]) - fhat_obs_j
          phi_sum <- phi_sum + fhat_cf + omega[k] * R_k
          phi_cnt <- phi_cnt + 1L
        }
      }
    } else {
      # Fallback: non-cached path
      for (r in seq_along(object$forests)) {
        fs <- object$forests[[r]]
        for (dir in list(list(rf=fs$rfA, hon=fs$idxB), list(rf=fs$rfB, hon=fs$idxA))) {
          X_ord <- .get_X_ord(object, dir$rf)
          y_hon <- rep(NA_real_, n)
          y_hon[dir$hon] <- as.numeric(Y[dir$hon])
          preds_cf <- honest_predict_cpp(dir$rf$forest, X_cf, X_ord, y_hon, as.integer(dir$hon))
          preds_obs <- honest_predict_cpp(dir$rf$forest, X_ord, X_ord, y_hon, as.integer(dir$hon))
          for (j in seq_along(dir$hon)) {
            k <- dir$hon[j]
            if (is.na(preds_cf[k]) || is.na(preds_obs[k])) next
            R_k <- as.numeric(Y[k]) - preds_obs[k]
            phi_sum <- phi_sum + preds_cf[k] + omega[k] * R_k
            phi_cnt <- phi_cnt + 1L
          }
        }
      }
    }

    mu[q] <- if (phi_cnt > 0) phi_sum / phi_cnt else NA_real_
  }
  mu
}

#' @keywords internal
.predict_complete <- function(object, newdata, n, Y) {
  n_queries <- nrow(newdata)
  pred_sum <- numeric(n_queries); pred_cnt <- integer(n_queries)

  newdata_num <- newdata
  for (col in names(newdata_num)) {
    if (is.factor(newdata_num[[col]]))
      newdata_num[[col]] <- as.numeric(newdata_num[[col]])
  }

  caches <- object$forest_caches
  use_cached <- !is.null(caches)

  if (use_cached) {
    X_query <- reorder_X_to_ranger(newdata_num, object$forests[[1]]$rfA)
    for (cn in names(caches)) {
      preds <- honest_predict_cached_cpp(caches[[cn]], X_query)
      for (q in seq_len(n_queries)) {
        if (!is.na(preds[q])) {
          pred_sum[q] <- pred_sum[q] + preds[q]
          pred_cnt[q] <- pred_cnt[q] + 1L
        }
      }
    }
  } else {
    for (r in seq_along(object$forests)) {
      fs <- object$forests[[r]]
      for (dir in list(list(rf=fs$rfA, hon=fs$idxB), list(rf=fs$rfB, hon=fs$idxA))) {
        X_ord_hon <- .get_X_ord(object, dir$rf)
        y_hon <- rep(NA_real_, n)
        y_hon[dir$hon] <- as.numeric(Y[dir$hon])
        X_query <- reorder_X_to_ranger(newdata_num, dir$rf)
        preds <- honest_predict_cpp(dir$rf$forest, X_query, X_ord_hon,
                                     y_hon, as.integer(dir$hon))
        for (q in seq_len(n_queries)) {
          if (!is.na(preds[q])) { pred_sum[q] <- pred_sum[q] + preds[q]; pred_cnt[q] <- pred_cnt[q] + 1L }
        }
      }
    }
  }

  out <- rep(NA_real_, n_queries)
  valid <- pred_cnt > 0
  out[valid] <- pred_sum[valid] / pred_cnt[valid]
  out
}

#' @keywords internal
.marginalized_predict_single <- function(object, query_vals, specified_vars, n, Y,
                                          propensities) {
  phi_sum <- 0; phi_cnt <- 0L

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    for (dir in list(list(rf=fs$rfA, hon=fs$idxB), list(rf=fs$rfB, hon=fs$idxA))) {
      X_ord <- .get_X_ord(object, dir$rf)
      y_hon <- rep(NA_real_, n)
      y_hon[dir$hon] <- as.numeric(Y[dir$hon])
      hon <- dir$hon
      X_cf <- X_ord
      for (vname in specified_vars) {
        col_idx <- get_ranger_col_idx(dir$rf, vname)
        val <- query_vals[[vname]]
        if (is.factor(object$X[[vname]]))
          val <- as.numeric(factor(val, levels = levels(object$X[[vname]])))
        X_cf[, col_idx + 1] <- val
      }
      preds_cf <- honest_predict_cpp(dir$rf$forest, X_cf, X_ord, y_hon, as.integer(hon))
      preds_obs <- honest_predict_cpp(dir$rf$forest, X_ord, X_ord, y_hon, as.integer(hon))

      # Compute sigma2_ej for each continuous specified variable
      sigma2_map <- list()
      for (vname in specified_vars) {
        prop <- propensities[[vname]]
        if (!prop$is_binary) {
          x_v <- as.numeric(prop$x_var[hon])
          g_v <- prop$ghat[hon]
          sigma2_map[[vname]] <- mean((x_v - g_v)^2)
        }
      }

      for (j in seq_along(hon)) {
        k <- hon[j]
        fhat_cf <- preds_cf[k]; fhat_obs <- preds_obs[k]
        if (is.na(fhat_cf) || is.na(fhat_obs)) next
        R_k <- as.numeric(Y[k]) - fhat_obs

        # Sum propensity-weighted corrections across all specified variables
        total_omega <- 0
        for (vname in specified_vars) {
          prop <- propensities[[vname]]
          if (prop$is_binary) {
            x_jk <- as.numeric(prop$x_var[k])
            gc <- max(0.025, min(0.975, prop$ghat[k]))
            qv <- as.numeric(query_vals[[vname]])
            if (qv == 1) {
              total_omega <- total_omega + x_jk / gc
            } else {
              total_omega <- total_omega + (1 - x_jk) / (1 - gc)
            }
          } else {
            x_jk <- as.numeric(prop$x_var[k])
            ej <- x_jk - prop$ghat[k]
            total_omega <- total_omega + ej / sigma2_map[[vname]]
          }
        }

        phi_sum <- phi_sum + fhat_cf + total_omega * R_k
        phi_cnt <- phi_cnt + 1L
      }
    }
  }
  if (phi_cnt > 0) phi_sum / phi_cnt else NA_real_
}

#' @keywords internal
.estimate_sigma2 <- function(object, newdata) {
  n <- nrow(object$X); X <- object$X; Y <- as.numeric(object$Y); p <- ncol(X)
  .fit_mean <- function(idx_tr, idx_te, seed) {
    dat_tr <- X[idx_tr,,drop=FALSE]; dat_tr$y <- Y[idx_tr]
    rf <- inf.ranger::ranger(y~., data=dat_tr, num.trees=1000L, mtry=p,
                              min.node.size=1, sample.fraction=1.0, replace=FALSE,
                              seed=seed, num.threads=1L)
    as.numeric(.ranger_predict(rf, data=X[idx_te,,drop=FALSE])$predictions)
  }
  m1_sum <- numeric(n); m1_cnt <- integer(n)
  m2_sum <- numeric(n); m2_cnt <- integer(n)
  for (r in seq_len(5L)) {
    idxA <- sample.int(n, n%/%2); idxB <- setdiff(seq_len(n), idxA)
    m1_sum[idxB] <- m1_sum[idxB] + .fit_mean(idxA,idxB,r*100L+1L)
    m1_sum[idxA] <- m1_sum[idxA] + .fit_mean(idxB,idxA,r*100L+2L)
    m1_cnt[idxB] <- m1_cnt[idxB]+1L; m1_cnt[idxA] <- m1_cnt[idxA]+1L
    idxC <- sample.int(n, n%/%2); idxD <- setdiff(seq_len(n), idxC)
    m2_sum[idxD] <- m2_sum[idxD] + .fit_mean(idxC,idxD,r*100L+3L)
    m2_sum[idxC] <- m2_sum[idxC] + .fit_mean(idxD,idxC,r*100L+4L)
    m2_cnt[idxD] <- m2_cnt[idxD]+1L; m2_cnt[idxC] <- m2_cnt[idxC]+1L
  }
  mhat1 <- m1_sum/pmax(m1_cnt,1L); mhat2 <- m2_sum/pmax(m2_cnt,1L)
  sprod <- (Y-mhat1)*(Y-mhat2)
  dat_scale <- X; dat_scale$sprod <- sprod
  rf_scale <- inf.ranger::ranger(sprod~., data=dat_scale, num.trees=1500L,
                                  mtry=p, min.node.size=10L,
                                  sample.fraction=1.0, replace=TRUE, num.threads=1L)
  pmax(as.numeric(.ranger_predict(rf_scale,
    data=newdata[,colnames(X),drop=FALSE])$predictions), 1e-8)
}

#' @keywords internal
.get_full_forest_predictions <- function(object) {
  n <- nrow(object$X); pred_sum <- numeric(n); pred_cnt <- integer(n)
  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    for (dir in list(list(rf=fs$rfA, hon=fs$idxB), list(rf=fs$rfB, hon=fs$idxA))) {
      X_ord <- .get_X_ord(object, dir$rf)
      y_hon <- rep(NA_real_, n)
      y_hon[dir$hon] <- as.numeric(object$Y[dir$hon])
      preds <- honest_predict_cpp(dir$rf$forest, X_ord, X_ord, y_hon, as.integer(dir$hon))
      for (k in dir$hon) {
        if (!is.na(preds[k])) { pred_sum[k] <- pred_sum[k]+preds[k]; pred_cnt[k] <- pred_cnt[k]+1L }
      }
    }
  }
  out <- rep(NA_real_, n); valid <- pred_cnt > 0
  out[valid] <- pred_sum[valid] / pred_cnt[valid]; out
}


#' @export
print.infForest_prediction <- function(x, ...) {
  cat("Inference Forest Predictions\n")
  pt <- if ("pred_type" %in% names(x)) x$pred_type[1] else "unknown"
  if (!is.null(pt) && !is.na(pt) && pt != "unknown")
    cat("  Type:", if (pt == "pointwise") "Pointwise" else "Marginalized", "\n")
  if ("marginalized_over" %in% names(x) && nchar(x$marginalized_over[1]) > 0)
    cat("  Marginalized over:", x$marginalized_over[1], "\n")
  cat("  Queries:", nrow(x), "\n\n")
  print_df <- x
  print_df$marginalized_over <- NULL
  print_df$pred_type <- NULL
  class(print_df) <- "data.frame"
  print(print_df, digits = 4, row.names = FALSE)
  invisible(x)
}
