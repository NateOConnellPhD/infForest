#' Fit an Inference Forest
#'
#' Fits a random forest with the inference forest design: standardized splitting
#' criterion, softmax variable selection, and honest cross-fitted estimation.
#' The returned object supports effect estimation, effect curves, interactions,
#' resolution analysis, and PASR-based confidence intervals.
#'
#' @param formula Object of class \code{formula} (e.g., \code{y ~ .}).
#' @param data Data frame containing the variables in \code{formula}.
#' @param num.trees Number of trees per forest. Default 5000.
#' @param mtry Number of candidate variables per split. Default \code{floor(sqrt(p))}.
#' @param min.node.size Minimum terminal node size. Default 10.
#' @param sample.fraction Fraction of observations per tree. Default 1.0
#'   (full sample).
#' @param replace Logical; sample with replacement? Default \code{FALSE}.
#'   Bootstrap resampling (\code{TRUE}) reduces effective sample size per tree
#'   to ~63\% of the training fold, which compounds with the honesty split to
#'   yield ~0.63 * n/2 unique observations per tree. Subsampling without
#'   replacement (\code{FALSE}) uses the full training fold.
#' @param honesty Logical; use honest cross-fitting? Default \code{TRUE}.
#'   When \code{TRUE}, data are split into build and estimation folds.
#' @param honesty.splits Number of independent fold assignments to average
#'   over. Default 5. Higher values reduce fold-assignment variance.
#' @param penalize Logical; use standardized splitting criterion? Default
#'   \code{TRUE}. Corrects balance and search advantages.
#' @param softmax Logical; use softmax (proportional) variable selection?
#'   Default \code{FALSE}. When \code{TRUE}, variables are selected with
#'   probability proportional to their standardized criterion rather than
#'   by argmax. Requires \code{penalize = TRUE}.
#' @param probability Logical; fit a probability forest for binary outcomes?
#'   Automatically set to \code{TRUE} when the response is a factor with 2
#'   levels.
#' @param num.threads Number of threads for ranger. Default 1.
#' @param seed Random seed for reproducibility. Default \code{NULL}.
#' @param verbose Logical; print progress? Default \code{FALSE}.
#'
#' @return An object of class \code{infForest} containing:
#' \describe{
#'   \item{forests}{List of fitted ranger forest objects (one per fold direction,
#'     per honesty split).}
#'   \item{X}{The predictor data frame.}
#'   \item{Y}{The response vector (numeric).}
#'   \item{fold_assignments}{List of fold assignments used for honest estimation.}
#'   \item{outcome_type}{Character: \code{"continuous"} or \code{"binary"}.}
#'   \item{call}{The matched call.}
#'   \item{params}{List of all fitting parameters.}
#' }
#'
#' @examples
#' \dontrun{
#' dat <- data.frame(y = rnorm(200), x1 = rnorm(200),
#'                   x2 = rbinom(200, 1, 0.4), x3 = rnorm(200))
#' fit <- infForest(y ~ ., data = dat)
#' effect(fit, "x2")
#' }
#'
#' @export
infForest <- function(formula,
                      data,
                      num.trees = 5000L,
                      mtry = NULL,
                      min.node.size = 10L,
                      sample.fraction = 1.0,
                      replace = FALSE,
                      honesty = TRUE,
                      honesty.splits = 5L,
                      penalize = TRUE,
                      softmax = FALSE,
                      probability = NULL,
                      num.threads = 1L,
                      seed = NULL,
                      verbose = FALSE) {

  cl <- match.call()

  # --- Parse formula ---
  mf <- model.frame(formula, data, na.action = na.pass)
  Y_raw <- model.response(mf)
  X <- data[, setdiff(names(data), as.character(formula[[2]])), drop = FALSE]
  n <- nrow(X)
  p <- ncol(X)

  # --- Detect outcome type ---
  if (is.factor(Y_raw) && nlevels(Y_raw) == 2) {
    outcome_type <- "binary"
    Y <- as.numeric(Y_raw) - 1  # 0/1
    if (is.null(probability)) probability <- TRUE
  } else if (is.numeric(Y_raw)) {
    outcome_type <- "continuous"
    Y <- Y_raw
    if (is.null(probability)) probability <- FALSE
  } else {
    stop("Response must be numeric (continuous) or a 2-level factor (binary).")
  }

  # --- Defaults ---
  if (is.null(mtry)) mtry <- floor(sqrt(p))
  if (!is.null(seed)) set.seed(seed)

  if (replace && honesty) {
    warning("replace = TRUE with honesty = TRUE reduces effective sample size per tree to ~63% of each fold. ",
            "With n/2 observations per fold, each tree uses only ~0.63 * n/2 unique observations for structure. ",
            "Consider replace = FALSE (the default) for honest estimation.")
  }

  X_num <- as.matrix(X)

  # --- Build ranger arg template ---
  build_ranger_args <- function(train_data, tree_seed) {
    args <- list(
      formula = y ~ .,
      data = train_data,
      num.trees = num.trees,
      mtry = mtry,
      min.node.size = min.node.size,
      sample.fraction = sample.fraction,
      replace = replace,
      num.threads = num.threads,
      write.forest = TRUE,
      seed = tree_seed,
      penalize.split.competition = penalize,
      softmax.split = softmax
    )
    if (probability) args$probability <- TRUE
    args
  }

  # --- Fit forests ---
  if (honesty) {
    # Cross-fitted honest estimation
    forests <- list()
    fold_list <- list()

    for (r in seq_len(honesty.splits)) {
      # Random fold assignment
      fold_seed <- if (!is.null(seed)) seed * 13L + r * 1000L else sample.int(.Machine$integer.max, 1)
      set.seed(fold_seed)
      fold <- sample(rep(1:2, length.out = n))
      fold_list[[r]] <- fold

      idxA <- which(fold == 1)
      idxB <- which(fold == 2)

      # Forest A: build on fold A
      datA <- X[idxA, , drop = FALSE]
      if (outcome_type == "continuous") {
        datA$y <- Y[idxA]
      } else {
        datA$y <- factor(Y[idxA], levels = c(0, 1))
      }
      seedA <- if (!is.null(seed)) seed + r * 100L + 1L else NULL
      rfA <- do.call(inf.ranger::ranger, build_ranger_args(datA, seedA))

      # Forest B: build on fold B
      datB <- X[idxB, , drop = FALSE]
      if (outcome_type == "continuous") {
        datB$y <- Y[idxB]
      } else {
        datB$y <- factor(Y[idxB], levels = c(0, 1))
      }
      seedB <- if (!is.null(seed)) seed + r * 100L + 2L else NULL
      rfB <- do.call(inf.ranger::ranger, build_ranger_args(datB, seedB))

      forests[[r]] <- list(
        rfA = rfA, rfB = rfB,
        idxA = idxA, idxB = idxB,
        fold = fold
      )
    }
  } else {
    # Standard (non-honest) forest — single fit
    dat_full <- X
    if (outcome_type == "continuous") {
      dat_full$y <- Y
    } else {
      dat_full$y <- factor(Y, levels = c(0, 1))
    }
    rf_full <- do.call(inf.ranger::ranger, build_ranger_args(dat_full, seed))
    forests <- list(list(rf = rf_full))
    fold_list <- NULL
  }

  # --- Build return object ---
  out <- list(
    forests = forests,
    X = X,
    X_num = X_num,
    Y = Y,
    fold_assignments = fold_list,
    outcome_type = outcome_type,
    honesty = honesty,
    honesty.splits = if (honesty) honesty.splits else 0L,
    call = cl,
    params = list(
      num.trees = num.trees,
      mtry = mtry,
      min.node.size = min.node.size,
      sample.fraction = sample.fraction,
      replace = replace,
      penalize = penalize,
      softmax = softmax,
      probability = probability,
      num.threads = num.threads
    )
  )
  class(out) <- "infForest"
  out
}


#' Print method for infForest objects
#'
#' @param x An \code{infForest} object.
#' @param ... Additional arguments (ignored).
#' @export
print.infForest <- function(x, ...) {
  cat("Inference Forest\n")
  cat("  Outcome type:    ", x$outcome_type, "\n")
  cat("  Observations:    ", nrow(x$X), "\n")
  cat("  Predictors:      ", ncol(x$X), "\n")
  cat("  Trees per forest:", x$params$num.trees, "\n")
  cat("  mtry:            ", x$params$mtry, "\n")
  cat("  min.node.size:   ", x$params$min.node.size, "\n")
  cat("  Honest:          ", x$honesty, "\n")
  if (x$honesty) {
    cat("  Honesty splits:  ", x$honesty.splits, "\n")
  }
  cat("  Penalized splits:", x$params$penalize, "\n")
  cat("  Softmax splits:  ", x$params$softmax, "\n")
  invisible(x)
}
