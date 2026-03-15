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
