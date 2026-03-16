#' Estimate the Nonlinear Effect Curve for a Continuous Predictor
#'
#' Constructs the effect curve tracing how the population-averaged outcome
#' varies with a continuous predictor, using honest within-leaf local slope
#' estimation. The curve captures nonlinear and non-monotone relationships
#' without functional form assumptions. Any contrast between two points on
#' the curve recovers the effect estimate from \code{effect()}.
#'
#' @param object An \code{infForest} object fitted with \code{honesty = TRUE}.
#' @param var Character; name of a continuous predictor variable.
#' @param q_lo,q_hi Quantiles defining the grid bounds. Default 0.10 and 0.90.
#' @param bw Bandwidth: target number of honest observations per grid interval.
#'   Controls grid density. Higher values = smoother curve. Default 20.
#' @param ref Reference value for the curve (curve is zero at this point).
#'   Default: median of the variable.
#' @param ... Additional arguments (currently unused).
#'
#' @return A list of class \code{infForest_curve} containing:
#' \describe{
#'   \item{variable}{Variable name.}
#'   \item{grid}{Grid points at which the curve is evaluated.}
#'   \item{curve}{Curve values (relative to reference).}
#'   \item{slopes}{Per-interval local slopes.}
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
                                   bw = 20L, ref = NULL, ...) {

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
  grid <- seq(lo, hi, length.out = n_intervals + 1)

  # Collect slopes from all honesty splits
  all_slopes <- matrix(0, nrow = object$honesty.splits, ncol = n_intervals)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]

    slopes_AB <- .extract_curve_slopes(fs$rfA, object$X, object$Y,
                                       honest_idx = fs$idxB, var = var,
                                       grid = grid)
    slopes_BA <- .extract_curve_slopes(fs$rfB, object$X, object$Y,
                                       honest_idx = fs$idxA, var = var,
                                       grid = grid)
    all_slopes[r, ] <- (slopes_AB + slopes_BA) / 2
  }

  avg_slopes <- colMeans(all_slopes)
  intervals <- diff(grid)
  curve_vals <- c(0, cumsum(avg_slopes * intervals))

  # Shift so curve = 0 at reference
  ref_val <- approx(grid, curve_vals, xout = ref, rule = 2)$y
  curve_vals <- curve_vals - ref_val

  out <- list(
    variable = var,
    grid = grid,
    curve = curve_vals,
    slopes = avg_slopes,
    intervals = intervals,
    ref = ref,
    n_intervals = n_intervals,
    q_lo = q_lo,
    q_hi = q_hi
  )
  class(out) <- "infForest_curve"
  out
}


#' @keywords internal
.extract_curve_slopes <- function(rf, X, Y, honest_idx, var, grid) {
  X_ord <- reorder_X_to_ranger(X, rf)
  col_idx <- get_ranger_col_idx(rf, var)

  n <- nrow(X)
  y_hon <- rep(NA_real_, n)
  y_hon[honest_idx] <- as.numeric(Y[honest_idx])

  win <- .grid_to_windows(grid)

  # Precompute augmentation: predict with X_j at median
  mid_val <- median(X_ord[, col_idx + 1L])
  X_ref <- X_ord
  X_ref[, col_idx + 1L] <- mid_val
  fhat_ref <- predict(rf, data = as.data.frame(X_ref))$predictions

  res <- honest_curve(
    rf$forest, X_ord, y_hon, as.integer(honest_idx),
    col = col_idx,
    midpoints = win$midpts,
    window_lo = win$wlo,
    window_hi = win$whi,
    fhat_ref_vec = fhat_ref
  )

  slopes <- res$popavg
  slopes[is.na(slopes)] <- 0
  slopes
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
