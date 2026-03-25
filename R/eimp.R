#' Effect Importance
#'
#' Computes the standardized criterion importance \eqn{\bar{\Delta}_j} for each
#' variable — the average excess impurity reduction beyond the EVT-corrected
#' null floor, across all nodes where the variable was a splitting candidate.
#' This is the nonparametric analogue of the partial F-statistic in regression.
#'
#' For interactions, computes the attenuation factor \eqn{\lambda_{jk}} that
#' quantifies how much interaction signal is lost due to partial splitting.
#' The corrected interaction estimate is \eqn{\hat{\psi}_{int} / \hat{\lambda}}.
#'
#' Variables with \eqn{\bar{\Delta}_j \leq 0} are below the noise floor and
#' can be safely removed. The AIPW inference in the refitted model preserves
#' type I error because the sandwich recalibrates to the new estimator's variance.
#'
#' @param object A fitted \code{infForest} object (requires \code{penalize = TRUE}).
#' @param interactions Character vector of interaction terms to evaluate, e.g.
#'   \code{c("x2:trt", "x1:x3")}. Default \code{NULL} skips interaction diagnostics.
#' @param vif_max Maximum variance inflation factor for interaction estimability.
#'   Default 10.
#'
#' @return An object of class \code{infForest_eimp} containing:
#' \describe{
#'   \item{main}{Data frame of main effect importance with columns: variable,
#'     delta_bar, n_nodes, pi (split frequency), signal (TRUE if delta_bar > 0).}
#'   \item{interactions}{Data frame of interaction diagnostics (if requested) with
#'     columns: term, pi_j, pi_k, pi_jk, lambda, vif, estimable.}
#' }
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat, num.trees = 5000, penalize = TRUE)
#' eimp(fit)
#' eimp(fit, interactions = c("x2:trt"))
#' plot(eimp(fit))
#' }
#'
#' @export
eimp <- function(object, interactions = NULL, vif_max = 10) {
  if (!inherits(object, "infForest"))
    stop("eimp() requires a fitted infForest object.")

  # --- Main effect importance: delta_bar_j ---
  # Requires standardized criterion sums stored during fitting.
  # These are stored in each ranger sub-forest as:
  #   rf$forest$criterion.sums  — numeric vector of length p, sum of delta_tilde_j
  #   rf$forest$criterion.counts — integer vector of length p, number of nodes evaluated

  vn <- names(object$X)
  p <- length(vn)
  delta_sum <- numeric(p)
  delta_cnt <- integer(p)

  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    for (rf in list(fs$rfA, fs$rfB)) {
      if (is.null(rf$forest$criterion.sums)) {
        stop("Standardized criterion values not stored. Refit with penalize = TRUE ",
             "using the updated inf.ranger that stores criterion scores.")
      }
      delta_sum <- delta_sum + rf$forest$criterion.sums
      delta_cnt <- delta_cnt + rf$forest$criterion.counts
    }
  }

  delta_bar <- ifelse(delta_cnt > 0, delta_sum / delta_cnt, 0)

  # Split frequency (pi_j) from tree structure
  n_trees_total <- 0L
  split_counts <- integer(p)
  for (r in seq_along(object$forests)) {
    fs <- object$forests[[r]]
    for (rf in list(fs$rfA, fs$rfB)) {
      B <- rf$num.trees
      n_trees_total <- n_trees_total + B
      for (b in seq_len(B)) {
        sv <- rf$forest$split.varIDs[[b]]
        vars_in_tree <- unique(sv[sv >= 0])
        for (v in vars_in_tree) {
          split_counts[v + 1L] <- split_counts[v + 1L] + 1L
        }
      }
    }
  }
  pi_j <- split_counts / n_trees_total

  main_df <- data.frame(
    variable = vn,
    delta_bar = round(delta_bar, 6),
    n_nodes = delta_cnt,
    pi = round(pi_j, 4),
    signal = delta_bar > 0,
    stringsAsFactors = FALSE
  )
  main_df <- main_df[order(-main_df$delta_bar), ]
  rownames(main_df) <- NULL

  # --- Interaction diagnostics: lambda_jk ---
  int_df <- NULL
  if (!is.null(interactions)) {
    int_rows <- list()

    for (term in interactions) {
      parts <- strsplit(term, ":")[[1]]
      if (length(parts) != 2) {
        warning("Skipping '", term, "': interaction terms must be 'var1:var2'")
        next
      }
      var_j <- parts[1]; var_k <- parts[2]
      if (!var_j %in% vn || !var_k %in% vn) {
        warning("Skipping '", term, "': variable not found in model")
        next
      }

      col_j <- which(vn == var_j) - 1L  # 0-based
      col_k <- which(vn == var_k) - 1L

      # Compute pi_j, pi_k, pi_jk from tree structure
      n_j <- 0L; n_k <- 0L; n_jk <- 0L; n_total <- 0L
      for (r in seq_along(object$forests)) {
        fs <- object$forests[[r]]
        for (rf in list(fs$rfA, fs$rfB)) {
          for (b in seq_len(rf$num.trees)) {
            sv <- rf$forest$split.varIDs[[b]]
            has_j <- any(sv == col_j)
            has_k <- any(sv == col_k)
            n_total <- n_total + 1L
            if (has_j) n_j <- n_j + 1L
            if (has_k) n_k <- n_k + 1L
            if (has_j && has_k) n_jk <- n_jk + 1L
          }
        }
      }

      pi_j_hat <- n_j / n_total
      pi_k_hat <- n_k / n_total
      pi_jk_hat <- n_jk / n_total

      # Lambda computation
      # For binary var_k: p_k = prevalence of level 1
      # For continuous var_k: rho_leaf ≈ 1
      x_k <- object$X[[var_k]]
      if (is.factor(x_k) || length(unique(x_k)) == 2) {
        p_k <- mean(as.numeric(x_k != levels(factor(x_k))[1]))
        rho <- 1.0
      } else {
        p_k <- 0.5  # not used for continuous-continuous
        rho <- 1.0
      }

      lambda <- 1 - (pi_j_hat - pi_jk_hat) * p_k - (pi_k_hat - pi_jk_hat) * rho
      lambda <- max(lambda, 0)  # floor at 0
      vif <- if (lambda > 0) 1 / lambda^2 else Inf

      int_rows[[length(int_rows) + 1]] <- data.frame(
        term = term,
        pi_j = round(pi_j_hat, 4),
        pi_k = round(pi_k_hat, 4),
        pi_jk = round(pi_jk_hat, 4),
        lambda = round(lambda, 4),
        vif = round(vif, 2),
        estimable = vif < vif_max,
        stringsAsFactors = FALSE
      )
    }

    if (length(int_rows) > 0) {
      int_df <- do.call(rbind, int_rows)
      rownames(int_df) <- NULL
    }
  }

  out <- list(main = main_df, interactions = int_df,
              n_trees = n_trees_total, vif_max = vif_max)
  class(out) <- "infForest_eimp"
  out
}


#' @method print infForest_eimp
#' @export
print.infForest_eimp <- function(x, ...) {
  cat("Effect Importance (", x$n_trees, " trees)\n\n", sep = "")

  cat("Main effects:\n")
  m <- x$main
  lw <- max(nchar(m$variable))
  for (i in seq_len(nrow(m))) {
    flag <- if (m$signal[i]) "  *" else ""
    cat(sprintf("  %-*s  delta_bar = %8.4f  pi = %.2f%s\n",
                lw, m$variable[i], m$delta_bar[i], m$pi[i], flag))
  }
  cat("\n  * = above noise floor (delta_bar > 0)\n")

  n_signal <- sum(m$signal)
  n_noise <- sum(!m$signal)
  if (n_noise > 0) {
    cat(sprintf("\n  %d signal, %d noise. Refit without: %s\n",
                n_signal, n_noise,
                paste(m$variable[!m$signal], collapse = ", ")))
  }

  if (!is.null(x$interactions)) {
    cat("\nInteractions:\n")
    for (i in seq_len(nrow(x$interactions))) {
      r <- x$interactions[i, ]
      est_flag <- if (r$estimable) "estimable" else "NOT estimable"
      cat(sprintf("  %s  lambda = %.3f  VIF = %.1f  (%s)\n",
                  r$term, r$lambda, r$vif, est_flag))
      cat(sprintf("    pi_j = %.3f  pi_k = %.3f  pi_jk = %.3f\n",
                  r$pi_j, r$pi_k, r$pi_jk))
    }
  }

  invisible(x)
}


#' @method plot infForest_eimp
#' @export
plot.infForest_eimp <- function(x, ...) {
  m <- x$main
  m <- m[order(m$delta_bar), ]  # ascending so highest at top
  p <- nrow(m)

  cols <- ifelse(m$delta_bar > 0, "steelblue", "gray70")

  par(mar = c(4, max(nchar(m$variable)) * 0.6 + 1, 2, 3))
  bp <- barplot(m$delta_bar, horiz = TRUE, names.arg = m$variable,
                las = 1, col = cols, border = NA,
                xlab = expression(bar(Delta)[j]),
                main = "Effect importance", ...)

  # Zero line
  abline(v = 0, col = "red", lty = 2)

  # Overlay delta_bar values
  text(pmax(m$delta_bar, 0) + max(abs(m$delta_bar)) * 0.03, bp,
       labels = sprintf("%.4f", m$delta_bar), adj = 0, cex = 0.75)

  invisible(x)
}
