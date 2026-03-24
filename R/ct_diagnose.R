#' Diagnose Covariance Floor
#'
#' Estimates the pointwise covariance floor C_T(x) across prediction points
#' using PASR. Returns summary statistics and supports visualization of where
#' prediction uncertainty is highest across the covariate space.
#'
#' The covariance floor is the irreducible variance component of the forest
#' prediction at each point — the variance that persists even with infinite
#' trees. It reflects structural dependence between trees induced by the
#' forest design acting on the realized covariate configuration.
#'
#' This is a diagnostic for deployed ranger models. It can inform hyperparameter
#' tuning: designs with lower median C_T achieve tighter prediction intervals.
#'
#' @param object A fitted \code{ranger} object.
#' @param data Training data frame. Required because ranger does not store
#'   training data.
#' @param newdata Data frame of prediction points. Default \code{NULL} uses
#'   training data.
#' @param R_max Maximum PASR replicates. Default 150.
#' @param B_mc Trees per paired forest. Default 500.
#' @param alpha Significance level. Default 0.05.
#' @param verbose Print progress. Default FALSE.
#' @param ... Additional arguments passed to \code{pasr_predict}.
#'
#' @return An object of class \code{infForest_ct} containing:
#' \describe{
#'   \item{Ct}{Numeric vector of C_T(x) estimates at each prediction point.}
#'   \item{mc_var}{Numeric vector of Monte Carlo variance V/B at each point.}
#'   \item{f_hat}{Numeric vector of forest predictions at each point.}
#'   \item{total_var}{Numeric vector of total variance (mc_var + Ct).}
#'   \item{newdata}{The prediction data frame.}
#'   \item{summary}{Named list of summary statistics.}
#' }
#'
#' @examples
#' \dontrun{
#' rf <- ranger(y ~ ., data = dat_train, num.trees = 5000)
#' ct <- ct_diagnose(rf, data = dat_train)
#' ct
#' plot(ct)
#' plot(ct, by = "x1")
#' }
#'
#' @export
ct_diagnose <- function(object, newdata = NULL, data = NULL, ...) {

  if (inherits(object, "pasr_ranger")) {
    # Use existing fitted object — no refitting
    vn <- object$vn
    if (is.null(newdata) && !is.null(data)) {
      newdata <- data[, vn, drop = FALSE]
    } else if (is.null(newdata)) {
      stop("newdata is required when passing a pasr_ranger object without data.")
    }
    pasr_res <- predict(object, newdata = newdata, unconditional = FALSE)
    outcome_type <- object$outcome_type
  } else if (inherits(object, "ranger")) {
    stop("ct_diagnose() requires a fitted pasr_ranger object from pasr_predict(). ",
         "Run ps <- pasr_predict(rf, data = ...) first, then ct_diagnose(ps).")
  } else {
    stop("ct_diagnose requires a pasr_ranger object from pasr_predict().")
  }

  summ <- list(
    mean   = mean(pasr_res$Ct_hat),
    median = median(pasr_res$Ct_hat),
    sd     = sd(pasr_res$Ct_hat),
    max    = max(pasr_res$Ct_hat),
    min    = min(pasr_res$Ct_hat),
    q25    = unname(quantile(pasr_res$Ct_hat, 0.25)),
    q75    = unname(quantile(pasr_res$Ct_hat, 0.75)),
    n_points = nrow(newdata)
  )

  out <- list(
    Ct = pasr_res$Ct_hat,
    mc_var = pasr_res$mc_var,
    f_hat = pasr_res$f_hat,
    total_var = pasr_res$mc_var + pasr_res$Ct_hat,
    newdata = newdata,
    summary = summ,
    outcome_type = outcome_type
  )
  class(out) <- "infForest_ct"
  out
}


#' Print method for infForest_ct objects
#'
#' @param x An \code{infForest_ct} object.
#' @param ... Additional arguments (ignored).
#' @export
print.infForest_ct <- function(x, ...) {
  cat("Covariance Floor Diagnostic (C_T)\n")
  cat("  Prediction points:", x$summary$n_points, "\n\n")
  cat("  C_T summary:\n")
  cat(sprintf("    Mean:   %.6f\n", x$summary$mean))
  cat(sprintf("    Median: %.6f\n", x$summary$median))
  cat(sprintf("    SD:     %.6f\n", x$summary$sd))
  cat(sprintf("    Range:  [%.6f, %.6f]\n", x$summary$min, x$summary$max))
  cat(sprintf("    IQR:    [%.6f, %.6f]\n", x$summary$q25, x$summary$q75))
  cat(sprintf("\n  MC variance (V/B) mean: %.6f\n", mean(x$mc_var)))
  cat(sprintf("  Ct / total variance:    %.1f%%\n",
              100 * x$summary$mean / mean(x$total_var)))
  invisible(x)
}


#' Plot method for infForest_ct objects
#'
#' @param x An \code{infForest_ct} object.
#' @param by Optional character; name of a covariate to plot C_T against.
#'   If \code{NULL}, plots C_T vs fitted value.
#' @param log_scale Logical; use log scale for C_T axis? Default \code{FALSE}.
#' @param ... Additional arguments passed to \code{plot}.
#' @export
plot.infForest_ct <- function(x, by = NULL, log_scale = FALSE, ...) {
  ct_vals <- x$Ct
  if (log_scale) ct_vals <- log10(pmax(ct_vals, 1e-10))
  ylab <- if (log_scale) expression(log[10](C[T](x))) else expression(C[T](x))

  if (is.null(by)) {
    plot(x$f_hat, ct_vals, xlab = expression(hat(f)(x)), ylab = ylab,
         main = "Covariance Floor vs Fitted Value",
         pch = 16, col = rgb(0, 0, 0, 0.3), ...)
  } else {
    if (!by %in% names(x$newdata)) {
      stop(paste0("Variable '", by, "' not found in prediction data."))
    }
    xvals <- x$newdata[[by]]
    plot(xvals, ct_vals, xlab = by, ylab = ylab,
         main = paste("Covariance Floor vs", by),
         pch = 16, col = rgb(0, 0, 0, 0.3), ...)
  }

  abline(h = if (log_scale) log10(x$summary$median) else x$summary$median,
         col = "red", lty = 2)
  legend("topright", legend = sprintf("Median C_T = %.2e", x$summary$median),
         col = "red", lty = 2, bty = "n", cex = 0.8)
}
