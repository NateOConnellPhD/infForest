#' Summary of Inference Forest Effects
#'
#' Computes and displays effect estimates for one or more predictors. Supports
#' a formula interface with optional bracket notation to specify comparison
#' points and interactions.
#'
#' For main effects, brackets specify comparison points:
#' \code{~ x1[0.10, 0.50, 0.90] + x2}
#'
#' For interactions, use \code{:} syntax where the first variable is the focal
#' effect and the second is the conditioning variable:
#' \code{~ x1:x2 + x1:x3[0.10,0.25,0.75,0.90]}
#'
#' Brackets on the focal variable (first) define \code{at} comparison points.
#' Brackets on the conditioning variable (second) define subgroup bands as
#' consecutive pairs. Use \code{*} as shorthand for main effects plus
#' interaction: \code{~ x1*x2} expands to \code{~ x1 + x2 + x1:x2}.
#'
#' @param object An \code{infForest} object.
#' @param vars Formula specifying which variables and interactions to summarize.
#'   Required. Use \code{summary(fit)} without vars for usage instructions.
#' @param type How to interpret bracketed values: \code{"quantile"} (default)
#'   or \code{"value"}. Can also be a named list for per-variable control.
#' @param bw Bandwidth for continuous effect estimation. Default 20.
#' @param q_lo,q_hi Grid bounds for slope estimation. Default 0.10 and 0.90.
#' @param ... Additional arguments (currently unused).
#'
#' @return A list of class \code{infForest_summary} containing effect and
#'   interaction results.
#'
#' @examples
#' \dontrun{
#' fit <- infForest(y ~ ., data = dat)
#' summary(fit, ~ x1 + x2)
#' summary(fit, ~ x1*x2)
#' summary(fit, ~ x1[0.10, 0.90]:x2)
#' summary(fit, ~ x1:x3[0.10,0.25,0.75,0.90])
#' }
#'
#' @export
summary.infForest <- function(object, vars = NULL, type = "quantile",
                              bw = 20L, q_lo = 0.10, q_hi = 0.90,
                              variance = c("sandwich", "pasr", "both"),
                              ci = TRUE, alpha = 0.05, ...) {

  check_infForest(object)
  variance <- match.arg(variance)


  if (is.null(vars)) {
    cat("Inference Forest Summary\n")
    cat("  Outcome type:    ", object$outcome_type, "\n")
    cat("  Observations:    ", nrow(object$X), "\n")
    cat("  Predictors:      ", ncol(object$X), "\n")
    cat("  Trees per forest:", object$params$num.trees, "\n")
    cat("  Honest splits:   ", object$honesty.splits, "\n\n")
    cat("Specify variables with a formula to compute effects:\n")
    cat("  summary(fit, ~ x1 + x2)\n")
    cat("  summary(fit, ~ x1[0.10, 0.50, 0.90] + x2)\n")
    cat("  summary(fit, ~ x1*x2)                           # main effects + interaction\n")
    cat("  summary(fit, ~ x1:x2)                           # interaction only\n")
    cat("  summary(fit, ~ x1:x3[0.10,0.25,0.75,0.90])     # with conditioning bands\n\n")
    cat("Available predictors:", paste(names(object$X), collapse = ", "), "\n")
    return(invisible(NULL))
  }

  specs <- .parse_summary_formula(vars)

  # Resolve per-variable type
  if (is.list(type)) {
    type_map <- type
  } else {
    type_map <- NULL
    type_default <- match.arg(type, c("quantile", "value"))
  }

  .get_type <- function(varname) {
    if (!is.null(type_map) && varname %in% names(type_map)) {
      return(type_map[[varname]])
    }
    if (!is.null(type_map)) return("quantile")
    type_default
  }

  results <- list()

  # Separate main effects into binary (batchable) and continuous (individual)
  binary_main_vars <- character(0)
  cont_specs <- list()
  interaction_specs <- list()

  for (spec in specs) {
    if (spec$kind == "main") {
      v <- spec$var
      check_varname(object, v)
      var_type <- detect_var_type(object$X[[v]])
      if (var_type == "binary") {
        binary_main_vars <- c(binary_main_vars, v)
      } else {
        cont_specs <- c(cont_specs, list(spec))
      }
    } else if (spec$kind == "interaction") {
      interaction_specs <- c(interaction_specs, list(spec))
    }
  }

  # Batch all binary main effects
  if (length(binary_main_vars) > 0) {
    for (v in binary_main_vars) {
      results[[v]] <- effect(object, v, variance = variance, ci = ci,
                             alpha = alpha, bw = bw)
    }
  }

  # Continuous main effects (each needs its own grid)
  for (spec in cont_specs) {
    v <- spec$var
    v_type <- .get_type(v)
    at <- if (!is.null(spec$at)) spec$at else c(0.25, 0.75)
    results[[v]] <- effect(object, v, at = at, type = v_type,
                           bw = bw, q_lo = q_lo, q_hi = q_hi,
                           variance = variance, ci = ci, alpha = alpha)
  }

  # Interactions
  for (spec in interaction_specs) {
    focal <- spec$focal
    by_var <- spec$by
    check_varname(object, focal)
    check_varname(object, by_var)

    label <- paste0(focal, ":", by_var)
    focal_at <- if (!is.null(spec$focal_at)) spec$focal_at else c(0.25, 0.75)
    focal_type <- .get_type(focal)

    by_var_type <- detect_var_type(object$X[[by_var]])

    if (by_var_type == "binary") {
      results[[label]] <- int(object, focal, by = by_var,
                              at = focal_at, type = focal_type,
                              bw = bw, q_lo = q_lo, q_hi = q_hi,
                              variance = variance, ci = ci, alpha = alpha)
    } else {
      by_at <- if (!is.null(spec$by_at)) spec$by_at else list(c(0.10, 0.25), c(0.75, 0.90))
      results[[label]] <- int(object, focal, by = by_var,
                              at = focal_at, type = focal_type,
                              by_at = by_at,
                              bw = bw, q_lo = q_lo, q_hi = q_hi,
                              variance = variance, ci = ci, alpha = alpha)
    }
  }

  out <- list(
    effects = results,
    n_terms = length(results)
  )
  class(out) <- "infForest_summary"
  out
}


#' @keywords internal
.parse_summary_formula <- function(f) {
  if (!inherits(f, "formula")) {
    stop("vars must be a formula (e.g., ~ x1 + x2 + x1:x2).")
  }

  rhs <- deparse(f[[2]], width.cutoff = 500)
  rhs <- paste(rhs, collapse = "")
  rhs <- gsub("\\s+", "", rhs)

  # Expand * terms: x1*x2 -> x1+x2+x1:x2
  # Handle brackets: x1[...]*x2[...] -> x1[...]+x2[...]+x1[...]:x2[...]
  rhs <- .expand_star_terms(rhs)

  # Split on + (but not inside brackets)
  terms <- .split_on_plus(rhs)

  # Parse each term
  specs <- list()
  seen <- character(0)  # deduplicate from * expansion
  for (term in terms) {
    parsed <- .parse_summary_term(term)
    # Dedup key
    key <- if (parsed$kind == "main") parsed$var else paste0(parsed$focal, ":", parsed$by)
    if (!key %in% seen) {
      specs <- c(specs, list(parsed))
      seen <- c(seen, key)
    }
  }
  specs
}


#' @keywords internal
.expand_star_terms <- function(rhs) {
  # Find A*B patterns (where A and B may contain brackets)
  # and expand to A+B+A:B
  result <- ""
  i <- 1
  chars <- strsplit(rhs, "")[[1]]
  n <- length(chars)

  tokens <- .tokenize_terms(rhs)

  # Look for * between tokens
  expanded <- character(0)
  k <- 1
  while (k <= length(tokens)) {
    if (k + 2 <= length(tokens) && tokens[k + 1] == "*") {
      A <- tokens[k]
      B <- tokens[k + 2]
      expanded <- c(expanded, A, "+", B, "+", paste0(A, ":", B))
      k <- k + 3
    } else {
      expanded <- c(expanded, tokens[k])
      k <- k + 1
    }
  }
  paste(expanded, collapse = "")
}


#' @keywords internal
.tokenize_terms <- function(rhs) {
  # Split into tokens: variable names (with optional brackets) and operators (+, *, :)
  tokens <- character(0)
  current <- ""
  depth <- 0
  chars <- strsplit(rhs, "")[[1]]

  for (ch in chars) {
    if (ch == "[") {
      depth <- depth + 1
      current <- paste0(current, ch)
    } else if (ch == "]") {
      depth <- depth - 1
      current <- paste0(current, ch)
    } else if (ch %in% c("+", "*") && depth == 0) {
      if (nchar(current) > 0) tokens <- c(tokens, current)
      tokens <- c(tokens, ch)
      current <- ""
    } else if (ch == ":" && depth == 0) {
      # Keep : attached — don't split here, it's part of interaction
      current <- paste0(current, ch)
    } else {
      current <- paste0(current, ch)
    }
  }
  if (nchar(current) > 0) tokens <- c(tokens, current)
  tokens
}


#' @keywords internal
.split_on_plus <- function(rhs) {
  terms <- character(0)
  depth <- 0
  current <- ""
  for (ch in strsplit(rhs, "")[[1]]) {
    if (ch == "[") {
      depth <- depth + 1
      current <- paste0(current, ch)
    } else if (ch == "]") {
      depth <- depth - 1
      current <- paste0(current, ch)
    } else if (ch == "+" && depth == 0) {
      terms <- c(terms, current)
      current <- ""
    } else {
      current <- paste0(current, ch)
    }
  }
  terms <- c(terms, current)
  terms <- trimws(terms)
  terms[nchar(terms) > 0]
}


#' @keywords internal
.parse_summary_term <- function(term) {
  # Check for interaction: contains : outside of brackets
  if (.has_colon_outside_brackets(term)) {
    # Split on : outside brackets
    parts <- .split_on_colon(term)
    if (length(parts) != 2) stop(paste0("Invalid interaction term: '", term, "'. Use var1:var2."))

    focal_parsed <- .parse_effect_term(parts[1])
    by_parsed <- .parse_effect_term(parts[2])

    # For the by variable, brackets define bands (consecutive pairs)
    by_at <- NULL
    if (!is.null(by_parsed$at)) {
      vals <- by_parsed$at
      if (length(vals) %% 2 != 0) {
        stop(paste0("Conditioning bands for '", by_parsed$var,
                    "' must have an even number of values (consecutive pairs define bands)."))
      }
      by_at <- lapply(seq(1, length(vals), by = 2), function(i) vals[i:(i+1)])
    }

    list(kind = "interaction",
         focal = focal_parsed$var,
         focal_at = focal_parsed$at,
         by = by_parsed$var,
         by_at = by_at)
  } else {
    # Main effect
    parsed <- .parse_effect_term(term)
    list(kind = "main", var = parsed$var, at = parsed$at)
  }
}


#' @keywords internal
.has_colon_outside_brackets <- function(term) {
  depth <- 0
  for (ch in strsplit(term, "")[[1]]) {
    if (ch == "[") depth <- depth + 1
    else if (ch == "]") depth <- depth - 1
    else if (ch == ":" && depth == 0) return(TRUE)
  }
  FALSE
}


#' @keywords internal
.split_on_colon <- function(term) {
  parts <- character(0)
  depth <- 0
  current <- ""
  for (ch in strsplit(term, "")[[1]]) {
    if (ch == "[") {
      depth <- depth + 1
      current <- paste0(current, ch)
    } else if (ch == "]") {
      depth <- depth - 1
      current <- paste0(current, ch)
    } else if (ch == ":" && depth == 0) {
      parts <- c(parts, current)
      current <- ""
    } else {
      current <- paste0(current, ch)
    }
  }
  parts <- c(parts, current)
  parts
}


#' @keywords internal
.parse_effect_term <- function(term) {
  if (grepl("\\[", term)) {
    var <- sub("\\[.*", "", term)
    inside <- sub(".*\\[(.*)\\]", "\\1", term)
    # Strip quotes and whitespace from each element
    raw <- strsplit(inside, ",")[[1]]
    raw <- trimws(gsub("[\"']", "", raw))
    # Try numeric first; if any fail, treat all as character (level names)
    at_num <- suppressWarnings(as.numeric(raw))
    if (any(is.na(at_num))) {
      at <- raw  # character level names
    } else {
      at <- at_num
    }
    list(var = var, at = at)
  } else {
    list(var = term, at = NULL)
  }
}


#' Print method for infForest_summary objects
#'
#' @param x An \code{infForest_summary} object.
#' @param ... Additional arguments (ignored).
#' @export
print.infForest_summary <- function(x, ...) {
  cat("Inference Forest Effect Summary\n")
  cat(paste(rep("-", 75), collapse = ""), "\n")

  for (nm in names(x$effects)) {
    eff <- x$effects[[nm]]

    if (inherits(eff, "infForest_interaction")) {
      cat(sprintf("  %s  (interaction: %s by %s)\n", nm, eff$variable, eff$by))
      for (k in seq_len(nrow(eff$subgroups))) {
        unit_label <- if (eff$var_type == "continuous") " (per unit)" else ""
        line <- sprintf("    %-35s  %8.4f%s",
                        eff$subgroups$subgroup[k], eff$subgroups$estimate[k], unit_label)
        if ("se" %in% names(eff$subgroups) && !is.na(eff$subgroups$se[k]))
          line <- paste0(line, sprintf("  (SE: %.4f)", eff$subgroups$se[k]))
        cat(line, "\n")
      }
      for (k in seq_len(nrow(eff$differences))) {
        line <- sprintf("    Diff: %-40s  %8.4f",
                        paste(eff$differences$hi[k], "-", eff$differences$lo[k]),
                        eff$differences$difference[k])
        if ("se" %in% names(eff$differences) && !is.na(eff$differences$se[k]))
          line <- paste0(line, sprintf("  (SE: %.4f, 95%% CI: [%.4f, %.4f])",
                                        eff$differences$se[k],
                                        eff$differences$ci_lower[k],
                                        eff$differences$ci_upper[k]))
        cat(line, "\n")
      }

    } else if (inherits(eff, "infForest_effect")) {
      if (eff$var_type == "binary") {
        line <- sprintf("  %-15s  binary      %8.4f", nm, eff$estimate)
        if (!is.null(eff$se))
          line <- paste0(line, sprintf("  (SE: %.4f, 95%% CI: [%.4f, %.4f])",
                                        eff$se, eff$ci_lower, eff$ci_upper))
        cat(line, "\n")
      } else {
        df <- eff$contrasts
        for (k in seq_len(nrow(df))) {
          label <- if (k == 1) sprintf("%-15s", nm) else sprintf("%-15s", "")
          line <- sprintf("  %s  %-16s  %8.4f  (per unit)", label, df$contrast[k], df$estimate[k])
          if ("se" %in% names(df) && !is.na(df$se[k]))
            line <- paste0(line, sprintf("  (SE: %.4f, 95%% CI: [%.4f, %.4f])",
                                          df$se[k], df$ci_lower[k], df$ci_upper[k]))
          cat(line, "\n")
        }
      }
    }
  }

  cat(paste(rep("-", 75), collapse = ""), "\n")
  invisible(x)
}
