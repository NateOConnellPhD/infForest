#' Split Frequency Diagnostic
#'
#' For each variable, reports the fraction of trees that split on it at least
#' once. Continuous variables split on multiple times in a single tree are
#' counted once. This is a diagnostic for the effective inclusion rate —
#' the probability that a variable participates in the forest's prediction.
#'
#' @param object A fitted \code{infForest} object.
#'
#' @return A data frame with columns:
#' \describe{
#'   \item{variable}{Variable name.}
#'   \item{n_trees}{Number of trees containing at least one split on this variable.}
#'   \item{pct}{Percentage of trees containing at least one split (0–100).}
#' }
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat, num.trees = 5000, penalize = TRUE)
#' split_frequency(fit)
#' }
#'
#' @export
split_frequency <- function(object) {
  if (!inherits(object, "infForest"))
    stop("split_frequency() requires a fitted infForest object.")

  vn <- names(object$X)
  p <- length(vn)

  # Total trees across all honesty splits × 2 forests per split
  counts <- integer(p)
  n_trees_total <- 0L

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    for (rf in list(fs$rfA, fs$rfB)) {
      B <- rf$num.trees
      n_trees_total <- n_trees_total + B
      # rf$forest$split.varIDs is a list of length B,
      # each element is an integer vector of variable indices (0-based)
      for (b in seq_len(B)) {
        sv <- rf$forest$split.varIDs[[b]]
        # unique variable indices that appear in this tree (exclude -1 for leaves)
        vars_in_tree <- unique(sv[sv >= 0])
        for (v in vars_in_tree) {
          counts[v + 1L] <- counts[v + 1L] + 1L  # 0-based to 1-based
        }
      }
    }
  }

  out <- data.frame(
    variable = vn,
    n_trees = counts,
    pct = round(100 * counts / n_trees_total, 1),
    stringsAsFactors = FALSE
  )
  out <- out[order(-out$pct), ]
  rownames(out) <- NULL
  class(out) <- c("infForest_split_freq", "data.frame")
  attr(out, "n_trees_total") <- n_trees_total
  out
}


#' @method print infForest_split_freq
#' @export
print.infForest_split_freq <- function(x, ...) {
  cat("Split Frequency (", attr(x, "n_trees_total"), " total trees)\n\n", sep = "")
  print.data.frame(x, row.names = FALSE)
  invisible(x)
}


#' @method plot infForest_split_freq
#' @export
plot.infForest_split_freq <- function(x, ...) {
  # Order by pct ascending so highest is at top of horizontal barplot
  x <- x[order(x$pct), ]
  p <- nrow(x)

  # Colors: gradient from light gray (low) to steelblue (high)
  cols <- colorRampPalette(c("gray80", "steelblue"))(p)

  par(mar = c(4, max(nchar(x$variable)) * 0.6 + 1, 2, 2))
  bp <- barplot(x$pct, horiz = TRUE, names.arg = x$variable,
                las = 1, col = cols, border = NA,
                xlab = "Trees containing variable (%)",
                main = "Split frequency", xlim = c(0, max(x$pct) * 1.15),
                ...)

  # Overlay percentages at end of each bar
  text(x$pct + max(x$pct) * 0.02, bp, labels = paste0(x$pct, "%"),
       adj = 0, cex = 0.8)

  invisible(x)
}
