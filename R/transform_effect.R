#' Transform Effect Estimates to Alternative Scales
#'
#' Converts adjusted mean differences (risk differences) to odds ratios,
#' risk ratios, or number needed to treat, with confidence intervals via
#' the delta method applied to per-observation AIPW influence function scores.
#'
#' Requires an \code{infForest_effect} or \code{infForest_means} object
#' with marginal means available (i.e., computed with \code{marginals = TRUE}
#' or via \code{forest_means()}).
#'
#' @param object An \code{infForest_effect} (with \code{marginals = TRUE})
#'   or \code{infForest_means} object for a binary predictor.
#' @param measure Character: \code{"OR"} (odds ratio), \code{"RR"} (risk ratio),
#'   \code{"RD"} (risk difference, the default from effect()), or \code{"NNT"}
#'   (number needed to treat). Default \code{"OR"}.
#' @param alpha Significance level. Default 0.05.
#' @param ... Currently unused.
#'
#' @return A list of class \code{infForest_transform} with elements:
#'   \code{measure}, \code{estimate}, \code{se}, \code{ci_lower}, \code{ci_upper},
#'   \code{log_estimate} (for OR/RR), \code{log_se}, and \code{df}.
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#' e <- effect(fit, "trt", marginals = TRUE)
#' transform_effect(e, "OR")
#' transform_effect(e, "RR")
#' transform_effect(e, "NNT")
#' }
#'
#' @export
transform_effect <- function(object, ...) UseMethod("transform_effect")

#' @rdname transform_effect
#' @export
transform_effect.infForest_effect <- function(object, measure = c("OR", "RR", "RD", "NNT"),
                                               alpha = 0.05, ...) {
  measure <- match.arg(measure)
  if (object$var_type != "binary")
    stop("transform_effect() requires a binary predictor.")
  if (is.null(object$marginal_means))
    stop("No marginal means. Rerun effect() with marginals = TRUE.")

  mm <- object$marginal_means
  # Level "1" is treatment, "0" is control
  idx_1 <- which(mm$level == "1")
  idx_0 <- which(mm$level == "0")
  if (length(idx_1) != 1 || length(idx_0) != 1)
    stop("Expected exactly two levels: '1' and '0'.")

  phi_1 <- attr(mm, "phi_1")
  phi_0 <- attr(mm, "phi_0")
  n_valid <- attr(mm, "n_valid")
  if (is.null(phi_1) || is.null(phi_0))
    stop("Per-observation scores not available. Recompute with forest_means() or effect(..., marginals = TRUE).")

  .do_transform(mm$mean[idx_1], mm$mean[idx_0],
                 phi_1, phi_0, n_valid,
                 measure, alpha, object$variable)
}

#' @rdname transform_effect
#' @export
transform_effect.infForest_means <- function(object, measure = c("OR", "RR", "RD", "NNT"),
                                              alpha = 0.05, ...) {
  measure <- match.arg(measure)
  if (object$var_type != "binary")
    stop("transform_effect() requires a binary predictor.")

  mm <- object$means
  idx_1 <- which(mm$level == "1")
  idx_0 <- which(mm$level == "0")
  if (length(idx_1) != 1 || length(idx_0) != 1)
    stop("Expected exactly two levels: '1' and '0'.")

  phi_1 <- attr(mm, "phi_1")
  phi_0 <- attr(mm, "phi_0")
  n_valid <- attr(mm, "n_valid")
  if (is.null(phi_1) || is.null(phi_0))
    stop("Per-observation scores not available. Recompute with forest_means().")

  .do_transform(mm$mean[idx_1], mm$mean[idx_0],
                 phi_1, phi_0, n_valid,
                 measure, alpha, object$variable)
}


# ============================================================
# Internal: generic delta method engine
# ============================================================
#' @keywords internal
.do_transform <- function(mu1, mu0, phi_1, phi_0, n_valid,
                            measure, alpha, variable) {
  z <- qnorm(1 - alpha / 2)

  if (measure == "RD") {
    est <- mu1 - mu0
    # Delta method scores: g(a,b) = a - b, g'1 = 1, g'2 = -1
    psi <- (phi_1 - mu1) - (phi_0 - mu0)
    se <- sd(psi) / sqrt(n_valid)
    out <- list(
      variable = variable, measure = "RD",
      estimate = est, se = se,
      ci_lower = est - z * se, ci_upper = est + z * se
    )

  } else if (measure == "RR") {
    # Work on log scale: g(a,b) = log(a) - log(b)
    # g'1 = 1/a, g'2 = -1/b
    log_rr <- log(mu1) - log(mu0)
    g1 <- 1 / mu1
    g2 <- -1 / mu0
    psi <- g1 * (phi_1 - mu1) + g2 * (phi_0 - mu0)
    se_log <- sd(psi) / sqrt(n_valid)
    out <- list(
      variable = variable, measure = "RR",
      estimate = exp(log_rr), se = NA_real_,
      ci_lower = exp(log_rr - z * se_log),
      ci_upper = exp(log_rr + z * se_log),
      log_estimate = log_rr, log_se = se_log
    )

  } else if (measure == "OR") {
    # Work on log scale: g(a,b) = log(a/(1-a)) - log(b/(1-b))
    # g'1 = 1/(a(1-a)), g'2 = -1/(b(1-b))
    log_or <- log(mu1 / (1 - mu1)) - log(mu0 / (1 - mu0))
    g1 <- 1 / (mu1 * (1 - mu1))
    g2 <- -1 / (mu0 * (1 - mu0))
    psi <- g1 * (phi_1 - mu1) + g2 * (phi_0 - mu0)
    se_log <- sd(psi) / sqrt(n_valid)
    out <- list(
      variable = variable, measure = "OR",
      estimate = exp(log_or), se = NA_real_,
      ci_lower = exp(log_or - z * se_log),
      ci_upper = exp(log_or + z * se_log),
      log_estimate = log_or, log_se = se_log
    )

  } else if (measure == "NNT") {
    # g(a,b) = 1/(a - b)
    # g'1 = -1/(a-b)^2, g'2 = 1/(a-b)^2
    rd <- mu1 - mu0
    nnt <- 1 / rd
    g1 <- -1 / rd^2
    g2 <- 1 / rd^2
    psi <- g1 * (phi_1 - mu1) + g2 * (phi_0 - mu0)
    se <- sd(psi) / sqrt(n_valid)
    out <- list(
      variable = variable, measure = "NNT",
      estimate = nnt, se = se,
      ci_lower = nnt - z * se, ci_upper = nnt + z * se
    )
  }

  out$alpha <- alpha

  # $df
  out$df <- data.frame(
    variable = variable,
    measure = measure,
    estimate = out$estimate,
    se = if (!is.na(out$se)) out$se else out$log_se,
    ci_lower = out$ci_lower,
    ci_upper = out$ci_upper,
    scale = if (measure %in% c("OR", "RR")) "log" else "natural",
    stringsAsFactors = FALSE
  )

  class(out) <- "infForest_transform"
  out
}


#' @export
print.infForest_transform <- function(x, ...) {
  cat("Inference Forest Effect Transformation\n")
  cat("  Variable:  ", x$variable, "\n")
  cat("  Measure:   ", x$measure, "\n")
  pct <- round((1 - x$alpha) * 100)

  if (x$measure %in% c("OR", "RR")) {
    cat(sprintf("  Estimate:   %.4f\n", x$estimate))
    cat(sprintf("  %d%% CI:     [%.4f, %.4f]\n", pct, x$ci_lower, x$ci_upper))
    cat(sprintf("  log scale:  %.4f  (SE = %.4f)\n", x$log_estimate, x$log_se))
  } else {
    cat(sprintf("  Estimate:   %.4f\n", x$estimate))
    cat(sprintf("  SE:         %.4f\n", x$se))
    cat(sprintf("  %d%% CI:     [%.4f, %.4f]\n", pct, x$ci_lower, x$ci_upper))
  }
  invisible(x)
}
