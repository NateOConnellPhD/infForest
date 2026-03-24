#' Adjusted Means from an Inference Forest
#'
#' Returns AIPW-adjusted means at specified covariate values, marginalized
#' over all unspecified variables. This is a convenience wrapper around
#' \code{predict()} with \code{pred_type = "marginal"}.
#'
#' For binary predictors, returns the adjusted mean at each level.
#' For continuous predictors, returns the adjusted mean at each query point.
#' For multiple variables, returns the joint marginalized mean.
#'
#' Can also extract marginal means from an existing \code{infForest_effect}
#' object computed with \code{marginals = TRUE}.
#'
#' Standard errors are computed via PASR. See the level_vs_contrast_variance
#' proof for why the sandwich is not valid for level-type estimands.
#'
#' @param object An \code{infForest} object or an \code{infForest_effect}
#'   result with marginals.
#' @param ... Named arguments specifying variable values. E.g.,
#'   \code{forest_means(fit, trt = c(0, 1))} or
#'   \code{forest_means(fit, trt = 1, x2 = 0.5)}.
#'   Alternatively, use \code{newdata} as a data frame.
#' @param newdata Data frame of query points (alternative to \code{...}).
#' @param alpha Significance level. Default 0.05.
#' @param R PASR replicates. Default 50.
#' @param verbose Print progress. Default FALSE.
#'
#' @return A data frame of class \code{infForest_means} with adjusted means,
#'   SEs, and CIs. Also contains \code{$df} in the universal format.
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#'
#' # Binary: adjusted means at each level
#' forest_means(fit, trt = c(0, 1))
#'
#' # Continuous: adjusted mean at specific values
#' forest_means(fit, x2 = c(-1, 0, 1))
#'
#' # Multiple variables: joint marginalized mean
#' forest_means(fit, trt = 1, x2 = 0.5)
#'
#' # From an effect object
#' e <- effect(fit, "trt", marginals = TRUE)
#' forest_means(e)
#' }
#'
#' @export
forest_means <- function(object, ...) UseMethod("forest_means")

#' @rdname forest_means
#' @export
forest_means.infForest_effect <- function(object, ...) {
  if (is.null(object$marginal_means))
    stop("No marginal means. Rerun effect() with marginals = TRUE.")

  mm <- object$marginal_means
  alpha <- if (!is.null(object$alpha)) object$alpha else 0.05
  z_crit <- qnorm(1 - alpha / 2)

  mm$ci_lower <- mm$mean - z_crit * mm$se
  mm$ci_upper <- mm$mean + z_crit * mm$se

  out <- list(
    variable = object$variable,
    var_type = object$var_type,
    means = mm,
    alpha = alpha
  )
  out$df <- .build_means_df(out)
  class(out) <- "infForest_means"
  out
}

#' @rdname forest_means
#' @export
forest_means.infForest <- function(object, ..., newdata = NULL,
                                    alpha = 0.05, R = 50L,
                                    verbose = FALSE) {
  # Build newdata from ... arguments if not provided
  if (is.null(newdata)) {
    dots_raw <- list(...)
    dots <- dots_raw[!names(dots_raw) %in% c("alpha", "R", "verbose", "newdata")]
    if (length(dots) == 0)
      stop("Specify variable values, e.g., forest_means(fit, trt = c(0, 1))")
    newdata <- expand.grid(dots, stringsAsFactors = FALSE)
  }

  result <- predict(object, newdata = newdata, pred_type = "marginal",
                     alpha = alpha, R = R, verbose = verbose)

  # Build means data frame
  specified_vars <- intersect(colnames(newdata), colnames(object$X))
  z_crit <- qnorm(1 - alpha / 2)

  mm <- data.frame(
    level = apply(newdata[, specified_vars, drop = FALSE], 1,
                  function(r) paste(names(r), "=", r, collapse = ", ")),
    mean = result$estimate,
    se = result$se,
    ci_lower = result$ci_lower,
    ci_upper = result$ci_upper,
    stringsAsFactors = FALSE
  )

  out <- list(
    variables = specified_vars,
    means = mm,
    alpha = alpha,
    prediction_result = result
  )
  out$df <- data.frame(
    variable = paste(specified_vars, collapse = " × "),
    type = "marginal",
    estimand = "mean",
    level = mm$level,
    estimate = mm$mean,
    se = mm$se,
    ci_lower = mm$ci_lower,
    ci_upper = mm$ci_upper,
    p.value = NA_real_,
    stringsAsFactors = FALSE
  )
  class(out) <- "infForest_means"
  out
}


#' @keywords internal
.build_means_df <- function(out) {
  mm <- out$means
  z_crit <- qnorm(1 - out$alpha / 2)
  data.frame(
    variable = out$variable,
    type = if (!is.null(out$var_type)) out$var_type else "marginal",
    estimand = "mean",
    level = as.character(mm$level),
    estimate = mm$mean,
    se = mm$se,
    ci_lower = mm$mean - z_crit * mm$se,
    ci_upper = mm$mean + z_crit * mm$se,
    p.value = NA_real_,
    stringsAsFactors = FALSE
  )
}


#' @export
print.infForest_means <- function(x, ...) {
  cat("Inference Forest Adjusted Means\n")
  if (!is.null(x$variable)) {
    cat("  Variable:   ", x$variable, "\n")
  } else if (!is.null(x$variables)) {
    cat("  Variables:  ", paste(x$variables, collapse = ", "), "\n")
  }
  cat("  Variance:    PASR\n\n")

  pct <- round((1 - x$alpha) * 100)
  mm <- x$means
  lw <- max(nchar(as.character(mm$level)))

  for (i in seq_len(nrow(mm))) {
    cat(sprintf("    %-*s  %.4f  SE = %.4f  %d%% CI [%.4f, %.4f]\n",
                lw, mm$level[i], mm$mean[i], mm$se[i], pct,
                mm$ci_lower[i], mm$ci_upper[i]))
  }
  invisible(x)
}
