#' @useDynLib infForest, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

#' @keywords internal
logit_fn <- function(z) 1 / (1 + exp(-z))

#' @keywords internal
.seed_int32 <- function(...) {
  key <- paste(..., sep = "|")
  b <- as.integer(charToRaw(key))
  h <- 0.0
  for (v in b) h <- (h * 131.0 + v) %% 2147483646.0
  s <- as.integer(h)
  if (is.na(s) || s <= 0L) s <- 1L
  s
}

#' @keywords internal
seed_from <- function(tag, scen, s = NA, r = NA, extra = NA) {
  .seed_int32("tag", tag, "scen", scen, "s", s, "r", r, "extra", extra)
}

#' @keywords internal
check_infForest <- function(object) {
  if (!inherits(object, "infForest")) {
    stop("Expected an infForest object. Fit one with infForest().")
  }
}

#' @keywords internal
check_varname <- function(object, var) {
  if (!var %in% names(object$X)) {
    stop(paste0("Variable '", var, "' not found in the data. Available: ",
                paste(names(object$X), collapse = ", ")))
  }
}

#' @keywords internal
detect_var_type <- function(x) {
  if (is.factor(x) || is.character(x)) return("categorical")
  ux <- unique(x[!is.na(x)])
  if (length(ux) <= 2 && all(ux %in% c(0, 1))) return("binary")
  "continuous"
}

#' @keywords internal
get_ranger_col_idx <- function(rf, varname) {
  # 0-based index in ranger's internal variable ordering
  vnames <- rf$forest$independent.variable.names
  idx <- match(varname, vnames)
  if (is.na(idx)) stop(paste0("Variable '", varname, "' not in forest."))
  as.integer(idx - 1L)
}

#' @keywords internal
reorder_X_to_ranger <- function(X, rf) {
  vnames <- rf$forest$independent.variable.names
  as.matrix(X[, vnames])
}

#' Extract predictions as numeric vector (handles both regression and probability forests)
#' @keywords internal
.get_pred_vector <- function(rf, newdata) {
  p <- predict(rf, data = newdata)$predictions
  if (is.matrix(p)) {
    # Probability forest: return P(Y=1) column
    return(p[, ncol(p)])
  }
  as.numeric(p)
}

#' Full nonparametric FWL residualization
#' Step 1: fit leave-X_j-out forest on (X_{-j}, Y) → h_hat → e_Y = Y - h_hat
#' Step 2: fit leave-X_j-out forest on (X_{-j}, X_j) → g_hat → e_j = X_j - g_hat
#' Returns list with Y_resid (e_Y for honest obs) and g_hat (for denominator correction)
#' @keywords internal
.residualize_FWL <- function(X, Y, build_idx, honest_idx, var) {
  X_minus_j <- X[, setdiff(names(X), var), drop = FALSE]
  n <- nrow(X)

  # Step 1: h_hat — predict Y from X_{-j}
  dat_h <- X_minus_j[build_idx, , drop = FALSE]
  dat_h$y <- as.numeric(Y[build_idx])
  rf_h <- ranger::ranger(y ~ ., data = dat_h, num.trees = 500,
                          mtry = min(5L, ncol(dat_h) - 1),
                          min.node.size = 5, seed = 42)
  h_hat <- predict(rf_h, data = X_minus_j)$predictions

  # Step 2: g_hat — predict X_j from X_{-j}
  x_j <- X[[var]]
  var_type <- detect_var_type(x_j)

  if (var_type == "binary") {
    dat_g <- X_minus_j[build_idx, , drop = FALSE]
    dat_g$xj <- factor(x_j[build_idx], levels = c(0, 1))
    rf_g <- ranger::ranger(xj ~ ., data = dat_g, num.trees = 500,
                            probability = TRUE,
                            mtry = min(5L, ncol(dat_g) - 1),
                            min.node.size = 5, seed = 43)
    g_hat <- predict(rf_g, data = X_minus_j)$predictions[, 2]  # P(X_j=1)
  } else {
    dat_g <- X_minus_j[build_idx, , drop = FALSE]
    dat_g$xj <- x_j[build_idx]
    rf_g <- ranger::ranger(xj ~ ., data = dat_g, num.trees = 500,
                            mtry = min(5L, ncol(dat_g) - 1),
                            min.node.size = 5, seed = 43)
    g_hat <- predict(rf_g, data = X_minus_j)$predictions
  }

  # e_Y = Y - h_hat (for honest obs only)
  Y_resid <- as.numeric(Y)
  Y_resid[honest_idx] <- Y_resid[honest_idx] - h_hat[honest_idx]

  # e_j = X_j - g_hat (for all obs — used in denominator)
  e_j <- x_j - g_hat

  list(Y_resid = Y_resid, e_j = e_j, g_hat = g_hat)
}

#' Simple wrapper for backward compatibility — returns just Y_resid
#' @keywords internal
.residualize_Y <- function(X, Y, build_idx, honest_idx, var) {
  res <- .residualize_FWL(X, Y, build_idx, honest_idx, var)
  res$Y_resid
}
