#' Variance Estimation for Effects (Deprecated)
#'
#' Use \code{\link{effect}} with the \code{variance} and \code{ci}
#' parameters instead.
#'
#' @param object An \code{infForest} object.
#' @param var Character; predictor name.
#' @param variance_method Character: "sandwich", "pasr", or "both".
#' @param ... Additional arguments passed to \code{effect()}.
#'
#' @return An \code{infForest_effect} object with SE and CI fields.
#'
#' @keywords internal
pasr_effect <- function(object, var, variance_method = "both", ...) {
  .Deprecated("effect",
    msg = "pasr_effect() is deprecated. Use effect(fit, var, variance = '...', ci = TRUE).")
  effect(object, var, variance = variance_method, ci = TRUE, ...)
}
