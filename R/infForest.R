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
#' @param fold_assignments Optional list of fold assignment vectors. If
#'   provided, these are used instead of generating random fold assignments.
#'   Each element is an integer vector of length n with values 1 or 2.
#'   Length of the list determines \code{honesty.splits}. Used internally
#'   by PASR to ensure paired forests share the same fold assignments.
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
                      honesty.splits = 5L,
                      penalize = TRUE,
                      softmax = FALSE,
                      probability = NULL,
                      num.threads = 1L,
                      seed = NULL,
                      verbose = FALSE,
                      fold_assignments = NULL) {

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

  if (replace) {
    warning("replace = TRUE reduces effective sample size per tree to ~63% of each fold. ",
            "With n/2 observations per fold, each tree uses only ~0.63 * n/2 unique observations for structure. ",
            "Consider replace = FALSE (the default).")
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

  # --- Fit forests (cross-fitted honest estimation) ---
  forests <- list()
  fold_list <- list()

  # Override honesty.splits if fold_assignments provided
  if (!is.null(fold_assignments)) {
    honesty.splits <- length(fold_assignments)
  }

  for (r in seq_len(honesty.splits)) {
    # Fold assignment: use provided or generate random
    if (!is.null(fold_assignments)) {
      fold <- fold_assignments[[r]]
    } else {
      fold_seed <- if (!is.null(seed)) seed * 13L + r * 1000L else sample.int(.Machine$integer.max, 1)
      set.seed(fold_seed)
      fold <- sample(rep(1:2, length.out = n))
    }
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

  # --- Cache ranger-ordered X matrix ---
  X_ord <- reorder_X_to_ranger(X, forests[[1]]$rfA)
  # Fix factor encoding for X_ord (ranger uses integer codes)
  X_df_clean <- X
  for (col in names(X_df_clean)) {
    if (is.factor(X_df_clean[[col]]) || is.character(X_df_clean[[col]])) {
      X_df_clean[[col]] <- as.numeric(as.factor(X_df_clean[[col]]))
    }
  }
  vnames <- forests[[1]]$rfA$forest$independent.variable.names
  X_ord <- as.matrix(X_df_clean[, vnames])

  # --- Precompute forest caches for fast scoring ---
  # One cache per (honesty_split, direction). Eliminates tree extraction,
  # obs_leaf routing, and leaf mean computation from every effect() call.
  forest_caches <- list()
  for (r in seq_along(forests)) {
    fs <- forests[[r]]
    # AB direction: forest A builds, fold B is honest
    y_hon_AB <- rep(NA_real_, n)
    y_hon_AB[fs$idxB] <- as.numeric(Y[fs$idxB])
    forest_caches[[paste0(r, "_AB")]] <- precompute_forest_cache_cpp(
      fs$rfA$forest, X_ord, y_hon_AB, as.integer(fs$idxB))
    # BA direction: forest B builds, fold A is honest
    y_hon_BA <- rep(NA_real_, n)
    y_hon_BA[fs$idxA] <- as.numeric(Y[fs$idxA])
    forest_caches[[paste0(r, "_BA")]] <- precompute_forest_cache_cpp(
      fs$rfB$forest, X_ord, y_hon_BA, as.integer(fs$idxA))
  }

  # --- Build return object ---
  out <- list(
    forests = forests,
    forest_caches = forest_caches,
    X = X,
    X_num = X_num,
    X_ord = X_ord,
    Y = Y,
    fold_assignments = fold_list,
    outcome_type = outcome_type,
    honesty.splits = honesty.splits,
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
  cat("  Honest:           yes\n")
  cat("  Honesty splits:  ", x$honesty.splits, "\n")
  cat("  Penalized splits:", x$params$penalize, "\n")
  cat("  Softmax splits:  ", x$params$softmax, "\n")
  invisible(x)
}
