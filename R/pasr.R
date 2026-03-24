#' Fit PASR Resampling Infrastructure
#'
#' Generates synthetic outcome draws and fits paired inference forests for
#' Procedure-Aligned Synthetic Resampling (PASR). Modifies the fitted
#' \code{infForest} object in place — stores cached PASR forests directly
#' on the object. All downstream functions automatically detect and use them.
#'
#' @param object A fitted \code{infForest} object.
#' @param R Number of PASR replicates. Default 100.
#' @param B Number of trees per PASR forest. Default 500. Does not need to
#'   match the deployed forest's \code{num.trees} — the covariance floor
#'   C_psi is independent of B, and V/B vanishes at B=500.
#' @param verbose Print progress. Default TRUE.
#' @param ... Additional arguments passed to \code{estimate_nuisance}.
#'
#' @return Invisible. The input object is modified in the calling environment
#'   with PASR caches stored in \code{object$pasr}.
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#' pasr(fit)                    # one-time cost, modifies fit in place
#'
#' # Everything just works
#' effect(fit, "trt", variance = "pasr")
#' effect_curve(fit, "x1", type = "level")
#' predict(fit, newdata = data.frame(trt = c(0, 1)))
#' forest_means(fit, trt = c(0, 1))
#' }
#'
#' @export
pasr <- function(object, R = 100L, B = 500L, verbose = TRUE, ...) {
  check_infForest(object)

  fit_name <- deparse(substitute(object))

  n <- nrow(object$X)
  X <- object$X
  Y <- object$Y

  # Step 1: Fit nuisance model
  if (verbose) cat("Fitting nuisance model...\n")
  nuisance <- estimate_nuisance(object, ...)

  # Step 2: Generate R synthetic Y draws
  if (verbose) cat("Generating", R, "synthetic outcome draws...\n")
  Y_syn_mat <- matrix(NA_real_, nrow = n, ncol = R)
  for (r in seq_len(R)) {
    Y_syn_mat[, r] <- generate_synthetic_Y(nuisance, seed = r * 7919L)
  }

  # Step 3: Fit R paired infForest models
  # Check for parallel backend
  has_future <- requireNamespace("future.apply", quietly = TRUE)
  use_parallel <- has_future && !inherits(future::plan(), "sequential")

  if (verbose) {
    if (use_parallel) {
      cat("Fitting", R, "paired forests in parallel...\n")
    } else {
      cat("Fitting", R, "paired forests sequentially...\n")
      if (!has_future)
        cat("  (install future.apply and set future::plan() for parallel)\n")
    }
  }

  # Build .fit_one_replicate in an isolated scope so it doesn't capture `object`
  .make_replicate_fn <- function(Y_syn_mat, X_local, X_ord_local, n_local,
                                  outcome_type_local, params_local, B_pasr_local) {
    function(r) {
      Y_syn <- Y_syn_mat[, r]

      dat_syn <- X_local
      if (outcome_type_local == "continuous") {
        dat_syn$y <- Y_syn
      } else {
        dat_syn$y <- factor(Y_syn, levels = c(0, 1))
      }

      set.seed(r * 5113L)
      fold <- sample(rep(1:2, length.out = n_local))
      idxA <- which(fold == 1)
      idxB <- which(fold == 2)

      dat_fA <- dat_syn[idxA, , drop = FALSE]
      dat_fB <- dat_syn[idxB, , drop = FALSE]

      .build_args <- function(dat, seed_val) {
        args <- list(
          formula = y ~ .,
          data = dat,
          num.trees = B_pasr_local,
          mtry = params_local$mtry,
          min.node.size = params_local$min.node.size,
          sample.fraction = params_local$sample.fraction,
          replace = params_local$replace,
          num.threads = 1L,
          write.forest = TRUE,
          seed = seed_val,
          penalize.split.competition = params_local$penalize,
          softmax.split = params_local$softmax
        )
        if (outcome_type_local == "binary") args$probability <- TRUE
        args
      }

      rfA1 <- do.call(inf.ranger::ranger, .build_args(dat_fA, r * 100L + 1L))
      rfA2 <- do.call(inf.ranger::ranger, .build_args(dat_fB, r * 100L + 1L))
      rfB1 <- do.call(inf.ranger::ranger, .build_args(dat_fA, r * 100L + 2L))
      rfB2 <- do.call(inf.ranger::ranger, .build_args(dat_fB, r * 100L + 2L))

      y_hon_AB <- rep(NA_real_, n_local)
      y_hon_AB[idxB] <- Y_syn[idxB]
      y_hon_BA <- rep(NA_real_, n_local)
      y_hon_BA[idxA] <- Y_syn[idxA]

      cacheA_AB <- precompute_forest_cache_cpp(rfA1$forest, X_ord_local, y_hon_AB, as.integer(idxB))
      cacheA_BA <- precompute_forest_cache_cpp(rfA2$forest, X_ord_local, y_hon_BA, as.integer(idxA))
      cacheB_AB <- precompute_forest_cache_cpp(rfB1$forest, X_ord_local, y_hon_AB, as.integer(idxB))
      cacheB_BA <- precompute_forest_cache_cpp(rfB2$forest, X_ord_local, y_hon_BA, as.integer(idxA))

      rm(rfA1, rfA2, rfB1, rfB2)

      list(
        caches = list(A_AB = cacheA_AB, A_BA = cacheA_BA,
                      B_AB = cacheB_AB, B_BA = cacheB_BA),
        fold = list(idxA = idxA, idxB = idxB)
      )
    }
  }

  .fit_one_replicate <- .make_replicate_fn(
    Y_syn_mat, X, object$X_ord, n,
    object$outcome_type, object$params, B
  )

  if (use_parallel) {
    results <- future.apply::future_lapply(
      seq_len(R), .fit_one_replicate,
      future.seed = TRUE,
      future.packages = "inf.ranger"
    )
  } else {
    results <- vector("list", R)
    for (r in seq_len(R)) {
      if (verbose && r %% 10 == 0) cat("  Replicate", r, "of", R, "\n")
      results[[r]] <- .fit_one_replicate(r)
    }
  }

  pasr_caches <- lapply(results, `[[`, "caches")
  fold_assignments <- lapply(results, `[[`, "fold")

  if (verbose) cat("PASR fitting complete.\n")

  pasr_obj <- list(
    caches = pasr_caches,
    Y_syn = Y_syn_mat,
    fold_assignments = fold_assignments,
    nuisance = nuisance,
    R = R,
    params = object$params,
    outcome_type = object$outcome_type,
    X = X,
    X_ord = object$X_ord
  )
  class(pasr_obj) <- "infForest_pasr"

  object$pasr <- pasr_obj
  assign(fit_name, object, envir = parent.frame())

  invisible(object)
}


#' @export
print.infForest_pasr <- function(x, ...) {
  cat("PASR Resampling Object\n")
  cat("  Replicates:  ", x$R, "\n")
  cat("  Outcome type:", x$outcome_type, "\n")
  cat("  Observations:", nrow(x$X), "\n")
  cat("  Trees/PASR forest:", x$caches[[1]]$A_AB$B, "\n")
  invisible(x)
}


# ============================================================
# Extract a scalar effect estimate from one PASR replicate
# Uses cached scorers — no ranger objects needed
# ============================================================
#' @keywords internal
.pasr_extract_effect <- function(pasr_obj, r, var, object,
                                  at, type, bw, q_lo, q_hi, ghat) {
  caches <- pasr_obj$caches[[r]]
  fa <- pasr_obj$fold_assignments[[r]]
  X <- pasr_obj$X
  n <- nrow(X)

  is_bin <- detect_var_type(object$X[[var]]) == "binary"
  col_idx <- get_ranger_col_idx(object$forests[[1]]$rfA, var)

  .extract_one_dir_cached <- function(cache) {
    nt <- .get_n_threads()
    if (is_bin) {
      res <- aipw_scores_cached_cpp(cache, ghat, col_idx, TRUE, 1, 0,
                                     n_threads = nt)
      return(res$psi)
    } else {
      x_var <- X[[var]]
      if (type == "quantile") {
        at_vals <- sort(unname(quantile(x_var, at)))
      } else {
        at_vals <- sort(at)
      }
      a <- at_vals[length(at_vals)]; b_val <- at_vals[1]
      grid_lo <- min(at_vals, unname(quantile(x_var, q_lo)))
      grid_hi <- max(at_vals, unname(quantile(x_var, q_hi)))
      n_honest <- n %/% 2
      n_intervals <- max(1L, as.integer(n_honest / bw))
      grid <- seq(grid_lo, grid_hi, length.out = n_intervals + 1)

      res <- aipw_curve_cached_cpp(cache, ghat, col_idx, grid,
                                    n_threads = nt)
      slopes <- res$slopes
      slopes[is.na(slopes)] <- 0
      curve_vals <- c(0, cumsum(slopes * diff(grid)))
      val_a <- approx(grid, curve_vals, xout = a, rule = 2)$y
      val_b <- approx(grid, curve_vals, xout = b_val, rule = 2)$y
      return((val_a - val_b) / (a - b_val))
    }
  }

  # Forest A: AB direction + BA direction, averaged
  est_A <- (.extract_one_dir_cached(caches$A_AB) +
             .extract_one_dir_cached(caches$A_BA)) / 2
  # Forest B: same
  est_B <- (.extract_one_dir_cached(caches$B_AB) +
             .extract_one_dir_cached(caches$B_BA)) / 2

  list(psi_A = est_A, psi_B = est_B)
}


# ============================================================
# Extract marginalized prediction from one PASR replicate
# Uses cached tree structure — no ranger objects needed
# ============================================================
#' @keywords internal
.pasr_extract_predict <- function(pasr_obj, r, object, newdata,
                                   specified_vars, propensities,
                                   precomputed = NULL) {
  caches <- pasr_obj$caches[[r]]
  fa <- pasr_obj$fold_assignments[[r]]
  Y_syn <- pasr_obj$Y_syn[, r]
  X_ord <- pasr_obj$X_ord
  n <- nrow(pasr_obj$X)
  n_queries <- nrow(newdata)
  is_complete <- length(specified_vars) == ncol(pasr_obj$X)

  nt <- .get_n_threads()

  if (is_complete) {
    pred_sum <- numeric(n_queries); pred_cnt <- integer(n_queries)
    X_query <- precomputed$X_query  # precomputed outside replicate loop
    for (cache in list(caches$A_AB, caches$A_BA)) {
      preds <- honest_predict_cached_cpp(cache, X_query, n_threads = nt)
      for (q in seq_len(n_queries)) {
        if (!is.na(preds[q])) {
          pred_sum[q] <- pred_sum[q] + preds[q]
          pred_cnt[q] <- pred_cnt[q] + 1L
        }
      }
    }
    out <- rep(NA_real_, n_queries)
    valid <- pred_cnt > 0
    out[valid] <- pred_sum[valid] / pred_cnt[valid]
    return(out)
  }

  # Marginalized: use precomputed X_cf matrices and omega weights
  mu <- numeric(n_queries)
  for (q in seq_len(n_queries)) {
    X_cf <- precomputed$X_cf_list[[q]]
    omega_vec <- precomputed$omega_list[[q]]
    phi_sum <- 0; phi_cnt <- 0L

    for (cache in list(caches$A_AB, caches$A_BA)) {
      hon <- as.integer(cache$honest_idx)
      fhat_obs_vec <- cache$fhat_obs

      # Counterfactual predictions via cached tree walk
      preds_cf <- honest_predict_cached_cpp(cache, X_cf, n_threads = nt)

      # Vectorized AIPW: phi_k = fhat_cf_k + omega_k * (Y_syn_k - fhat_obs_k)
      for (j in seq_along(hon)) {
        k <- hon[j]
        fhat_cf <- preds_cf[k]
        fhat_obs_j <- fhat_obs_vec[j]
        if (is.na(fhat_cf) || is.na(fhat_obs_j)) next
        R_k <- Y_syn[k] - fhat_obs_j
        phi_sum <- phi_sum + fhat_cf + omega_vec[k] * R_k
        phi_cnt <- phi_cnt + 1L
      }
    }
    mu[q] <- if (phi_cnt > 0) phi_sum / phi_cnt else NA_real_
  }
  mu
}


# ============================================================
# Precompute constants for predict extraction (called once)
# ============================================================
#' @keywords internal
.precompute_predict_constants <- function(object, newdata, specified_vars,
                                           propensities) {
  X_ord <- object$X_ord
  n <- nrow(object$X)
  n_queries <- nrow(newdata)
  is_complete <- length(specified_vars) == ncol(object$X)

  if (is_complete) {
    newdata_num <- newdata
    for (col in names(newdata_num))
      if (is.factor(newdata_num[[col]]))
        newdata_num[[col]] <- as.numeric(newdata_num[[col]])
    X_query <- reorder_X_to_ranger(newdata_num, object$forests[[1]]$rfA)
    return(list(X_query = X_query))
  }

  # Precompute col_idx once
  col_idxs <- list()
  for (vname in specified_vars)
    col_idxs[[vname]] <- get_ranger_col_idx(object$forests[[1]]$rfA, vname)

  # Precompute sigma2 for continuous variables
  sigma2_map <- list()
  for (vname in specified_vars) {
    prop <- propensities[[vname]]
    if (!prop$is_binary) {
      ej <- as.numeric(prop$x_var) - prop$ghat
      sigma2_map[[vname]] <- mean(ej^2)
    }
  }

  # Precompute X_cf matrices and omega vectors for each query point
  X_cf_list <- vector("list", n_queries)
  omega_list <- vector("list", n_queries)

  for (q in seq_len(n_queries)) {
    query_vals <- newdata[q, specified_vars, drop = FALSE]

    # Build counterfactual X
    X_cf <- X_ord
    for (vname in specified_vars) {
      val <- query_vals[[vname]]
      if (is.factor(object$X[[vname]]))
        val <- as.numeric(factor(val, levels = levels(object$X[[vname]])))
      X_cf[, col_idxs[[vname]] + 1] <- val
    }
    X_cf_list[[q]] <- X_cf

    # Build omega vector (length n, only honest obs used)
    omega <- numeric(n)
    for (vname in specified_vars) {
      prop <- propensities[[vname]]
      if (prop$is_binary) {
        x_j <- as.numeric(prop$x_var)
        gc <- pmax(0.025, pmin(0.975, prop$ghat))
        qv <- as.numeric(query_vals[[vname]])
        if (qv == 1) {
          omega <- omega + x_j / gc
        } else {
          omega <- omega + (1 - x_j) / (1 - gc)
        }
      } else {
        ej <- as.numeric(prop$x_var) - prop$ghat
        omega <- omega + ej / sigma2_map[[vname]]
      }
    }
    omega_list[[q]] <- omega
  }

  list(X_cf_list = X_cf_list, omega_list = omega_list)
}


# ============================================================
# Extract level curve from one PASR replicate
# Uses cached scorers
# ============================================================
#' @keywords internal
.pasr_extract_level_curve <- function(pasr_obj, r, object, var, grid, ghat, x_var) {
  caches <- pasr_obj$caches[[r]]
  fa <- pasr_obj$fold_assignments[[r]]
  Y_syn <- pasr_obj$Y_syn[, r]
  n <- nrow(pasr_obj$X)
  G1 <- length(grid)
  col_idx <- get_ranger_col_idx(object$forests[[1]]$rfA, var)

  mu_splits <- numeric(0)
  n_dirs <- 0L

  for (cache in list(caches$A_AB, caches$A_BA)) {
    hon <- as.integer(cache$honest_idx)
    res <- aipw_curve_cached_cpp(cache, ghat, col_idx, grid)
    fg <- res$fhat_grid
    fo <- cache$fhat_obs
    sigma2 <- res$sigma2_ej

    mu_g <- numeric(G1)
    for (g in seq_len(G1)) {
      phi_sum <- 0; cnt <- 0L
      for (j in seq_along(hon)) {
        k <- hon[j]
        fg_jg <- fg[j, g]; fo_j <- fo[j]
        if (is.na(fg_jg) || is.na(fo_j)) next
        omega_k <- (x_var[k] - ghat[k]) / sigma2
        R_k <- Y_syn[k] - fo_j
        phi_sum <- phi_sum + fg_jg + omega_k * R_k
        cnt <- cnt + 1L
      }
      mu_g[g] <- if (cnt > 0) phi_sum / cnt else NA_real_
    }

    if (n_dirs == 0L) { mu_splits <- mu_g } else { mu_splits <- mu_splits + mu_g }
    n_dirs <- n_dirs + 1L
  }

  mu_splits / n_dirs
}


# ============================================================
# Shared helper: parallel/sequential dispatch for PASR loops
# ============================================================
#' @keywords internal
.pasr_lapply <- function(R, FUN) {
  has_future <- requireNamespace("future.apply", quietly = TRUE)
  use_parallel <- has_future && !inherits(future::plan(), "sequential")

  if (use_parallel) {
    future.apply::future_lapply(seq_len(R), FUN, future.seed = TRUE)
  } else {
    lapply(seq_len(R), FUN)
  }
}


# ============================================================
# Get thread count from package option
# ============================================================
#' @keywords internal
.get_n_threads <- function() {
  getOption("infForest.threads", 1L)
}
