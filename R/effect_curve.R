#' Estimate the Nonlinear Effect Curve for a Continuous Predictor
#'
#' Constructs the effect curve tracing how the population-averaged outcome
#' varies with a continuous predictor, using honest AIPW estimation. The
#' curve captures nonlinear and non-monotone relationships without functional
#' form assumptions. Any contrast between two points on the curve recovers
#' the effect estimate from \code{effect()}.
#'
#' The estimator combines honest forest predictions at each grid point
#' (working model) with propensity-weighted honest residuals (debiasing
#' correction). The propensity correction is computed once and reused across
#' all grid intervals, since confounding bias is a property of the X_j-X_{-j}
#' correlation structure, not of where on X_j's support you evaluate.
#'
#' @param object An \code{infForest} object.
#' @param var Character; name of a continuous predictor variable.
#' @param q_lo,q_hi Quantiles defining the grid bounds. Default 0.10 and 0.90.
#' @param bw Bandwidth: target number of honest observations per grid interval.
#'   Controls grid density. Higher values = smoother curve. Default 20.
#' @param ref Reference value for the curve (curve is zero at this point).
#'   Default: median of the variable.
#' @param propensity_trees Number of trees for the propensity model. Default 2000.
#' @param ... Additional arguments (currently unused).
#'
#' @return A list of class \code{infForest_curve} containing:
#' \describe{
#'   \item{variable}{Variable name.}
#'   \item{grid}{Grid points at which the curve is evaluated.}
#'   \item{curve}{Curve values (relative to reference).}
#'   \item{slopes}{Per-interval AIPW-adjusted local slopes.}
#'   \item{ref}{Reference value.}
#'   \item{n_intervals}{Number of grid intervals used.}
#' }
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#' ec <- effect_curve(fit, "age")
#' plot(ec)
#' }
#'
#' @export
effect_curve <- function(object, ...) UseMethod("effect_curve")

#' @rdname effect_curve
#' @export
effect_curve.infForest <- function(object, var, q_lo = 0.10, q_hi = 0.90,
                                   bw = 20L, ref = NULL,
                                   propensity_trees = 2000L, ...) {

  check_infForest(object)
  check_varname(object, var)

  x_var <- object$X[[var]]
  if (detect_var_type(x_var) != "continuous") {
    stop("effect_curve() requires a continuous predictor. Use effect() for binary.")
  }

  if (is.null(ref)) ref <- unname(median(x_var))

  lo <- unname(quantile(x_var, q_lo))
  hi <- unname(quantile(x_var, q_hi))
  n_honest <- nrow(object$X) %/% 2
  n_intervals <- max(5L, min(20L, as.integer(n_honest / bw)))

  curve_result <- .aipw_build_curve(object, var, grid_lo = lo, grid_hi = hi,
                                     n_honest = n_honest, bw = bw,
                                     propensity_trees = propensity_trees)

  grid <- curve_result$grid
  curve_vals <- curve_result$curve

  # Shift so curve = 0 at reference
  ref_val <- approx(grid, curve_vals, xout = ref, rule = 2)$y
  curve_vals <- curve_vals - ref_val

  out <- list(
    variable = var,
    grid = grid,
    curve = curve_vals,
    slopes = curve_result$slopes,
    intervals = curve_result$intervals,
    ref = ref,
    n_intervals = n_intervals,
    q_lo = q_lo,
    q_hi = q_hi
  )
  class(out) <- "infForest_curve"
  out
}


#' Print method for infForest_curve objects
#'
#' @param x An \code{infForest_curve} object.
#' @param ... Additional arguments (ignored).
#' @export
print.infForest_curve <- function(x, ...) {
  cat("Inference Forest Effect Curve\n")
  cat("  Variable:   ", x$variable, "\n")
  cat("  Grid range: ", round(min(x$grid), 3), "to", round(max(x$grid), 3), "\n")
  cat("  Reference:  ", round(x$ref, 3), "\n")
  cat("  Intervals:  ", x$n_intervals, "\n")
  cat("  Curve range:", round(min(x$curve), 4), "to", round(max(x$curve), 4), "\n")
  invisible(x)
}


#' Plot method for infForest_curve objects
#'
#' @param x An \code{infForest_curve} object.
#' @param ... Additional arguments passed to \code{plot()}.
#' @export
plot.infForest_curve <- function(x, ...) {
  plot(x$grid, x$curve, type = "l", lwd = 2,
       xlab = x$variable, ylab = "Effect (relative to reference)",
       main = paste("Effect curve:", x$variable),
       ...)
  abline(h = 0, lty = 2, col = "gray50")
  abline(v = x$ref, lty = 3, col = "gray70")
}
