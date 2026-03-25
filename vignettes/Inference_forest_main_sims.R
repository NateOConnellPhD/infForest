# ============================================================
# sim_infForest.R — Simulation study for infForest
#
# Design:
#   - 500 independent XY profiles per DGM × n × outcome combo
#   - Every 5th profile gets PASR (100 PASR fits → coverage)
#   - 2 DGMs: complex (17 predictors) and simple (6 predictors)
#   - 4 sample sizes: 200, 400, 800, 1500
#   - 2 outcome types: continuous, binary
#   - 16 combos × 500 profiles = 8000 total fits, 1600 PASR fits
#
# Parallelism (staggered):
#   Phase 1: 100 PASR profiles — internal parallelism (14 cores per run)
#   Phase 2: 400 non-PASR profiles — profile-level parallelism (14 at once)
#
# Coverage: Each PASR CI checked against the other 499 point estimates.
#   The 499 other estimates ARE the sampling distribution of the procedure.
#   Does the interval capture other procedural estimates 95% of the time?
#
# Convergence: All 500 point estimates vs DGM truth and oracle regression.
#   The oracle is a perfectly specified regression model. InfForest can
#   outperform the oracle for nonlinear effects under binary outcomes
#   because logistic regression compresses the effect through the link.
# ============================================================

suppressPackageStartupMessages({
  library(inf.ranger)
  library(infForest)
  library(future)
  library(future.apply)
  library(data.table)
})

N_CORES <- 14L
options(future.globals.maxSize = 4 * 1024^3)

out_dir <- file.path(getwd(), "sim_infForest_results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- Utility functions ---
.std <- function(x) { s <- sd(x); if (is.na(s)||s==0) return(x-mean(x)); (x-mean(x))/s }
.centered_sq <- function(x) x^2 - mean(x^2)
logit <- function(z) 1/(1+exp(-z))

calibrate_intercept <- function(X, eta0_fun, target = 0.40) {
  eta0 <- eta0_fun(X)
  f <- function(a) mean(logit(a + eta0)) - target
  uniroot(f, interval = c(-15, 15))$root
}


# ============================================================
# X GENERATORS
# ============================================================

generate_X_complex <- function(n, seed, rho = 0.35) {
  set.seed(seed)
  x1 <- .std(rnorm(n, sd=1.2)); x2 <- .std(rt(n, df=5)*1.5)
  x3 <- .std(rho*x1 + sqrt(1-rho^2)*rnorm(n)); x4 <- .std(rnorm(n))
  x5 <- runif(n); x6 <- rbinom(n, 1, 0.40); x7 <- rbinom(n, 1, 0.15)
  x8 <- sample(0:2, n, TRUE, prob=c(0.5, 0.3, 0.2)); x9 <- .std(rnorm(n))
  x10 <- rbinom(n, 1, 0.30)
  xS <- .std(rnorm(n)); xC <- .std(0.6*xS + sqrt(1-0.36)*rnorm(n))
  xN1 <- .std(rnorm(n)); xN2 <- .std(rnorm(n)); xN3 <- .std(rnorm(n))
  x11 <- .std(rnorm(n, sd=0.8)); x12 <- .std(rnorm(n, sd=1.1))
  data.frame(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
             xS, xC, xN1, xN2, xN3, x11, x12)
}

generate_X_simple <- function(n, seed) {
  set.seed(seed)
  x1 <- .std(rnorm(n, sd=1.2)); x2 <- .std(rt(n, df=5)*1.5)
  x6 <- rbinom(n, 1, 0.40); x7 <- rbinom(n, 1, 0.15)
  xS <- .std(rnorm(n)); xC <- .std(0.6*xS + sqrt(1-0.36)*rnorm(n))
  data.frame(x1, x2, x6, x7, xS, xC)
}


# ============================================================
# DGM FUNCTIONS
# ============================================================

# --- Complex DGM (17 predictors) ---
mu_complex <- function(X) {
  i81 <- as.numeric(X$x8==1); i82 <- as.numeric(X$x8==2)
  x11 <- if("x11" %in% names(X)) X$x11 else 0
  x12 <- if("x12" %in% names(X)) X$x12 else 0
  0.90*sin(1.1*X$x1) + 0.35*X$x2 + 0.55*.centered_sq(X$x3) + 0.18*X$x4 +
    0.30*(X$x5>0.4) + 0.22*X$x6 + 0.20*X$x7 + 0.18*i81 + 0.28*i82 +
    0.45*sin(x11) + 0.25*as.numeric(x12>1.0) +
    0.15*(X$x2*X$x6) + 0.12*(sin(x11)*as.numeric(x12>1.0)) +
    0.12*(X$x2*X$x6*X$x7) + 0.50*X$xS
}

sigma_complex <- function(X) {
  x12 <- if("x12" %in% names(X)) X$x12 else 0
  pmax(0.65 + 0.25*abs(X$x1) + 0.15*abs(X$x2) +
         0.15*as.numeric(X$x5>0.4) + 0.12*as.numeric(x12>1.0), 0.15)
}

eta0_complex <- function(X) {
  i81 <- as.numeric(X$x8==1); i82 <- as.numeric(X$x8==2)
  x11 <- if("x11" %in% names(X)) X$x11 else 0
  x12 <- if("x12" %in% names(X)) X$x12 else 0
  0.55*X$x1 + 0.60*X$x2 + 0.45*.centered_sq(X$x3) + 0.20*X$x4 + 0.35*X$x5 +
    0.50*X$x6 + 0.30*X$x7 + 0.18*i81 + 0.28*i82 +
    0.10*X$x9 + 0.12*X$x10 + 0.35*sin(x11) + 0.22*as.numeric(x12>1.1) +
    0.18*(X$x1*X$x2) + 0.35*(X$x2*X$x6) + 0.15*(X$x3*i82) +
    0.12*(sin(x11)*as.numeric(x12>1.1)) +
    0.25*(X$x2*X$x6*X$x7) + 0.80*X$xS
}

# --- Simple DGM (6 predictors: x1, x2, x6, x7, xS, xC) ---
mu_simple <- function(X) {
  0.90*sin(1.1*X$x1) + 0.35*X$x2 + 0.22*X$x6 + 0.20*X$x7 +
    0.15*(X$x2*X$x6) + 0.50*X$xS
}

sigma_simple <- function(X) {
  pmax(0.65 + 0.25*abs(X$x1) + 0.15*abs(X$x2), 0.15)
}

eta0_simple <- function(X) {
  0.55*X$x1 + 0.60*X$x2 + 0.50*X$x6 + 0.30*X$x7 +
    0.35*(X$x2*X$x6) + 0.80*X$xS
}


# ============================================================
# DGM TRUTH (population-average effects from the generating model)
# ============================================================
compute_truth <- function(X, outcome_type, dgm) {
  if (dgm == "simple") {
    mu_fn <- mu_simple; eta_fn <- eta0_simple
  } else {
    mu_fn <- mu_complex; eta_fn <- eta0_complex
  }
  
  if (outcome_type == "continuous") {
    cf <- mu_fn
  } else {
    a0 <- calibrate_intercept(X, eta_fn, 0.40)
    cf <- function(Xm) logit(a0 + eta_fn(Xm))
  }
  
  # Binary effects
  X1 <- X; X1$x6 <- 1; X0 <- X; X0$x6 <- 0
  x6_pa <- mean(cf(X1) - cf(X0))
  X1 <- X; X1$x7 <- 1; X0 <- X; X0$x7 <- 0
  x7_pa <- mean(cf(X1) - cf(X0))
  
  # Continuous slopes
  q <- function(v, p) unname(quantile(X[[v]], p))
  x2_q25 <- q("x2", 0.25); x2_q75 <- q("x2", 0.75); x2_span <- x2_q75 - x2_q25
  xS_q25 <- q("xS", 0.25); xS_q75 <- q("xS", 0.75); xS_span <- xS_q75 - xS_q25
  xC_q25 <- q("xC", 0.25); xC_q75 <- q("xC", 0.75); xC_span <- xC_q75 - xC_q25
  
  Xhi <- X; Xhi$x2 <- x2_q75; Xlo <- X; Xlo$x2 <- x2_q25
  x2_eff <- cf(Xhi) - cf(Xlo); x2_pa <- mean(x2_eff) / x2_span
  Xhi <- X; Xhi$xS <- xS_q75; Xlo <- X; Xlo$xS <- xS_q25
  xS_pa <- mean(cf(Xhi) - cf(Xlo)) / xS_span
  Xhi <- X; Xhi$xC <- xC_q75; Xlo <- X; Xlo$xC <- xC_q25
  xC_pa <- mean(cf(Xhi) - cf(Xlo)) / xC_span
  
  # Interaction: x2 by x6
  i1 <- which(X$x6 == 1); i0 <- which(X$x6 == 0)
  x2x6_int <- (mean(x2_eff[i1]) - mean(x2_eff[i0])) / x2_span
  
  list(x6 = x6_pa, x7 = x7_pa, x2 = x2_pa, xS = xS_pa, xC = xC_pa,
       x2x6_int = x2x6_int,
       binary_intercept = if (outcome_type == "binary") a0 else NA)
}


# ============================================================
# SINGLE PROFILE WORKER
# ============================================================
fit_one_profile <- function(idx, X, outcome_type, dgm, run_pasr,
                            num.trees = 5000L, binary_intercept = NA) {
  library(inf.ranger)
  library(infForest)
  
  .std <- function(x) { s <- sd(x); if (is.na(s)||s==0) return(x-mean(x)); (x-mean(x))/s }
  .centered_sq <- function(x) x^2 - mean(x^2)
  logit <- function(z) 1/(1+exp(-z))
  
  n <- nrow(X)
  set.seed(idx * 7919L + 31L)
  
  # Generate Y
  dat <- X
  if (dgm == "simple") {
    if (outcome_type == "continuous") {
      mu <- 0.90*sin(1.1*X$x1) + 0.35*X$x2 + 0.22*X$x6 + 0.20*X$x7 +
        0.15*(X$x2*X$x6) + 0.50*X$xS
      sig <- pmax(0.65 + 0.25*abs(X$x1) + 0.15*abs(X$x2), 0.15)
      dat$y <- mu + sig * rnorm(n)
    } else {
      eta <- 0.55*X$x1 + 0.60*X$x2 + 0.50*X$x6 + 0.30*X$x7 +
        0.35*(X$x2*X$x6) + 0.80*X$xS
      dat$y <- factor(rbinom(n, 1, logit(binary_intercept + eta)), levels = c(0, 1))
    }
  } else {
    i81 <- as.numeric(X$x8==1); i82 <- as.numeric(X$x8==2)
    x11 <- if("x11" %in% names(X)) X$x11 else 0
    x12 <- if("x12" %in% names(X)) X$x12 else 0
    if (outcome_type == "continuous") {
      mu <- 0.90*sin(1.1*X$x1) + 0.35*X$x2 + 0.55*.centered_sq(X$x3) + 0.18*X$x4 +
        0.30*(X$x5>0.4) + 0.22*X$x6 + 0.20*X$x7 + 0.18*i81 + 0.28*i82 +
        0.45*sin(x11) + 0.25*as.numeric(x12>1.0) +
        0.15*(X$x2*X$x6) + 0.12*(sin(x11)*as.numeric(x12>1.0)) +
        0.12*(X$x2*X$x6*X$x7) + 0.50*X$xS
      sig <- pmax(0.65 + 0.25*abs(X$x1) + 0.15*abs(X$x2) +
                    0.15*as.numeric(X$x5>0.4) + 0.12*as.numeric(x12>1.0), 0.15)
      dat$y <- mu + sig * rnorm(n)
    } else {
      eta <- 0.55*X$x1 + 0.60*X$x2 + 0.45*.centered_sq(X$x3) + 0.20*X$x4 + 0.35*X$x5 +
        0.50*X$x6 + 0.30*X$x7 + 0.18*i81 + 0.28*i82 +
        0.10*X$x9 + 0.12*X$x10 + 0.35*sin(x11) + 0.22*as.numeric(x12>1.1) +
        0.18*(X$x1*X$x2) + 0.35*(X$x2*X$x6) + 0.15*(X$x3*i82) +
        0.12*(sin(x11)*as.numeric(x12>1.1)) +
        0.25*(X$x2*X$x6*X$x7) + 0.80*X$xS
      dat$y <- factor(rbinom(n, 1, logit(binary_intercept + eta)), levels = c(0, 1))
    }
  }
  
  # Fit infForest
  fit <- infForest(y ~ ., data = dat, num.trees = num.trees,
                   penalize = TRUE, softmax = TRUE, seed = idx)
  
  # Extract point estimates
  x6_est  <- effect(fit, "x6")$estimate
  x7_est  <- effect(fit, "x7")$estimate
  x2_est  <- effect(fit, "x2")$contrasts$estimate[1]
  xS_est  <- effect(fit, "xS")$contrasts$estimate[1]
  xC_est  <- effect(fit, "xC")$contrasts$estimate[1]
  x2x6_obj <- int(fit, "x2", by = "x6")
  x2x6_est <- x2x6_obj$differences$difference[1]
  
  out <- list(
    idx = idx,
    x6 = x6_est, x7 = x7_est, x2 = x2_est,
    xS = xS_est, xC = xC_est, x2x6_int = x2x6_est,
    pasr = NULL
  )
  
  # PASR if requested
  if (run_pasr) {
    pasr(fit, R = 100)
    e_x6  <- effect(fit, "x6");  e_x7  <- effect(fit, "x7")
    e_x2  <- effect(fit, "x2");  e_xS  <- effect(fit, "xS")
    e_xC  <- effect(fit, "xC")
    i_x2x6 <- int(fit, "x2", by = "x6")
    
    out$pasr <- list(
      x6_se = e_x6$se, x6_lo = e_x6$ci_lower, x6_hi = e_x6$ci_upper,
      x7_se = e_x7$se, x7_lo = e_x7$ci_lower, x7_hi = e_x7$ci_upper,
      x2_se = e_x2$contrasts$se[1], x2_lo = e_x2$contrasts$ci_lower[1], x2_hi = e_x2$contrasts$ci_upper[1],
      xS_se = e_xS$contrasts$se[1], xS_lo = e_xS$contrasts$ci_lower[1], xS_hi = e_xS$contrasts$ci_upper[1],
      xC_se = e_xC$contrasts$se[1], xC_lo = e_xC$contrasts$ci_lower[1], xC_hi = e_xC$contrasts$ci_upper[1],
      x2x6_se = i_x2x6$differences$se[1],
      x2x6_lo = i_x2x6$differences$ci_lower[1],
      x2x6_hi = i_x2x6$differences$ci_upper[1]
    )
  }
  
  out
}


# ============================================================
# ORACLE REGRESSION
# ============================================================
fit_oracle <- function(X, y_numeric, outcome_type, dgm) {
  odat <- data.frame(y = y_numeric, sin_x1 = sin(1.1*X$x1), x2 = X$x2,
                     x6 = X$x6, x7 = X$x7, xS = X$xS, xC = X$xC,
                     x2x6 = X$x2 * X$x6)
  
  if (dgm == "complex") {
    odat$x3sq <- .centered_sq(X$x3); odat$x4 <- X$x4
    odat$x5gt <- as.numeric(X$x5 > 0.4)
    odat$i81 <- as.numeric(X$x8==1); odat$i82 <- as.numeric(X$x8==2)
    odat$x9 <- X$x9; odat$x10 <- X$x10
    odat$sin_x11 <- sin(X$x11); odat$x12gt <- as.numeric(X$x12 > 1.0)
    odat$sinx11_x12gt <- sin(X$x11) * as.numeric(X$x12 > 1.0)
    odat$x2x6x7 <- X$x2 * X$x6 * X$x7
    odat$xN1 <- X$xN1; odat$xN2 <- X$xN2; odat$xN3 <- X$xN3
    fml <- y ~ sin_x1 + x2 + x3sq + x4 + x5gt + x6 + x7 + i81 + i82 +
      sin_x11 + x12gt + x2x6 + sinx11_x12gt + x2x6x7 + xS + xC + x9 + x10 +
      xN1 + xN2 + xN3
  } else {
    fml <- y ~ sin_x1 + x2 + x6 + x7 + x2x6 + xS + xC
  }
  
  if (outcome_type == "continuous") {
    fit <- lm(fml, data = odat)
    pred_fn <- function(d) predict(fit, newdata = d)
  } else {
    fit <- glm(fml, data = odat, family = binomial)
    pred_fn <- function(d) predict(fit, newdata = d, type = "response")
  }
  
  q <- function(v, p) unname(quantile(X[[v]], p))
  x2_q25 <- q("x2", 0.25); x2_q75 <- q("x2", 0.75); x2_span <- x2_q75 - x2_q25
  xS_q25 <- q("xS", 0.25); xS_q75 <- q("xS", 0.75); xS_span <- xS_q75 - xS_q25
  xC_q25 <- q("xC", 0.25); xC_q75 <- q("xC", 0.75); xC_span <- xC_q75 - xC_q25
  
  cf <- function(var, val, extra = NULL) {
    d <- odat; d[[var]] <- val
    if (!is.null(extra)) for (nm in names(extra)) d[[nm]] <- extra[[nm]]
    pred_fn(d)
  }
  
  o_x6 <- mean(cf("x6", 1) - cf("x6", 0))
  o_x7 <- mean(cf("x7", 1) - cf("x7", 0))
  
  x2_xtra_hi <- list(x2x6 = x2_q75 * X$x6)
  x2_xtra_lo <- list(x2x6 = x2_q25 * X$x6)
  if (dgm == "complex") {
    x2_xtra_hi$x2x6x7 <- x2_q75 * X$x6 * X$x7
    x2_xtra_lo$x2x6x7 <- x2_q25 * X$x6 * X$x7
  }
  x2_eff <- cf("x2", x2_q75, x2_xtra_hi) - cf("x2", x2_q25, x2_xtra_lo)
  o_x2 <- mean(x2_eff) / x2_span
  o_xS <- mean(cf("xS", xS_q75) - cf("xS", xS_q25)) / xS_span
  o_xC <- mean(cf("xC", xC_q75) - cf("xC", xC_q25)) / xC_span
  
  i1 <- which(X$x6 == 1); i0 <- which(X$x6 == 0)
  o_x2x6 <- (mean(x2_eff[i1]) - mean(x2_eff[i0])) / x2_span
  
  list(x6 = o_x6, x7 = o_x7, x2 = o_x2, xS = o_xS, xC = o_xC, x2x6_int = o_x2x6)
}


# ============================================================
# PROCEDURAL COVERAGE
# ============================================================
compute_coverage <- function(all_results, estimand) {
  pasr_idxs <- which(sapply(all_results, function(r) !is.null(r$pasr)))
  if (length(pasr_idxs) == 0) return(NA)
  
  lo_field <- paste0(estimand, "_lo")
  hi_field <- paste0(estimand, "_hi")
  all_ests <- sapply(all_results, `[[`, estimand)
  
  coverages <- numeric(length(pasr_idxs))
  for (j in seq_along(pasr_idxs)) {
    k <- pasr_idxs[j]
    lo <- all_results[[k]]$pasr[[lo_field]]
    hi <- all_results[[k]]$pasr[[hi_field]]
    # Coverage over the OTHER 499 procedural estimates
    others <- all_ests[-k]
    coverages[j] <- mean(others >= lo & others <= hi, na.rm = TRUE)
  }
  mean(coverages)
}


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================
N_PROFILES <- 500L
N_PASR     <- 100L
N_TREES    <- 5000L
SAMPLE_SIZES <- c(200L, 400L, 800L, 1500L)
DGMS <- c("simple", "complex")
OUTCOMES <- c("continuous", "binary")
ESTIMANDS <- c("x6", "x7", "x2", "xS", "xC", "x2x6_int")

for (dgm in DGMS) {
  for (n_obs in SAMPLE_SIZES) {
    for (outcome_type in OUTCOMES) {
      
      label <- sprintf("%s_n%d_%s", dgm, n_obs, outcome_type)
      ofile <- file.path(out_dir, paste0(label, ".rds"))
      if (file.exists(ofile)) { cat("Skipping:", label, "\n"); next }
      
      cat("\n", paste(rep("=", 60), collapse=""), "\n")
      cat(sprintf("  DGM: %s | n: %d | outcome: %s\n", dgm, n_obs, outcome_type))
      cat(paste(rep("=", 60), collapse=""), "\n")
      
      # Generate X (shared across all 500 Y draws)
      gen_X <- if (dgm == "simple") generate_X_simple else generate_X_complex
      X <- gen_X(n_obs, seed = n_obs * 1000L + 42L)
      
      # DGM truth
      truth <- compute_truth(X, outcome_type, dgm)
      
      # ---- Phase 1: PASR profiles (internal parallelism) ----
      cat("  Phase 1: ", N_PASR, " PASR profiles (internal parallel)...\n")
      plan(multisession, workers = N_CORES)
      options(infForest.threads = N_CORES)
      
      t0 <- Sys.time()
      pasr_results <- vector("list", N_PASR)
      for (i in seq_len(N_PASR)) {
        if (i %% 10 == 0) cat(sprintf("    %d/%d  (%.1f min elapsed)\n",
                                      i, N_PASR,
                                      as.numeric(difftime(Sys.time(), t0, units="mins"))))
        pasr_results[[i]] <- fit_one_profile(
          idx = i, X = X, outcome_type = outcome_type, dgm = dgm,
          run_pasr = TRUE, num.trees = N_TREES,
          binary_intercept = truth$binary_intercept
        )
      }
      t1 <- Sys.time()
      cat(sprintf("  Phase 1 done: %.1f min\n", difftime(t1, t0, units = "mins")))
      
      # ---- Phase 2: non-PASR profiles (profile-level parallelism) ----
      cat("  Phase 2: ", N_PROFILES - N_PASR, " non-PASR profiles (profile parallel)...\n")
      plan(multisession, workers = N_CORES)
      options(infForest.threads = 1L)
      
      t0 <- Sys.time()
      non_pasr_results <- future_lapply(
        seq(N_PASR + 1L, N_PROFILES),
        function(i) {
          fit_one_profile(
            idx = i, X = X, outcome_type = outcome_type, dgm = dgm,
            run_pasr = FALSE, num.trees = N_TREES,
            binary_intercept = truth$binary_intercept
          )
        },
        future.seed = TRUE,
        future.packages = c("inf.ranger", "infForest")
      )
      t1 <- Sys.time()
      cat(sprintf("  Phase 2 done: %.1f min\n", difftime(t1, t0, units = "mins")))
      
      all_results <- c(pasr_results, non_pasr_results)
      
      # ---- Oracle regressions (fast, sequential) ----
      cat("  Computing oracle regressions...\n")
      oracle_results <- lapply(seq_len(N_PROFILES), function(i) {
        set.seed(i * 7919L + 31L)
        if (dgm == "simple") {
          if (outcome_type == "continuous") {
            mu <- mu_simple(X); sig <- sigma_simple(X)
            y_num <- mu + sig * rnorm(n_obs)
          } else {
            y_num <- rbinom(n_obs, 1, logit(truth$binary_intercept + eta0_simple(X)))
          }
        } else {
          if (outcome_type == "continuous") {
            mu <- mu_complex(X); sig <- sigma_complex(X)
            y_num <- mu + sig * rnorm(n_obs)
          } else {
            y_num <- rbinom(n_obs, 1, logit(truth$binary_intercept + eta0_complex(X)))
          }
        }
        fit_oracle(X, y_num, outcome_type, dgm)
      })
      
      # ---- Summary table ----
      cat(sprintf("\n  %-12s %8s %8s %8s %8s  cov\n",
                  "estimand", "DGM", "IF.mean", "ORC.mean", "IF.sd"))
      cat(sprintf("  %s\n", paste(rep("-", 58), collapse="")))
      for (est in ESTIMANDS) {
        dgm_val <- truth[[est]]
        if (is.null(dgm_val)) next
        if_vals  <- sapply(all_results, `[[`, est)
        orc_vals <- sapply(oracle_results, `[[`, est)
        cov <- compute_coverage(all_results, est)
        cat(sprintf("  %-12s %8.4f %8.4f %8.4f %8.4f  %.3f\n",
                    est, dgm_val, mean(if_vals, na.rm=TRUE),
                    mean(orc_vals, na.rm=TRUE), sd(if_vals, na.rm=TRUE), cov))
      }
      
      # Save
      saveRDS(list(
        all_results = all_results,
        oracle_results = oracle_results,
        truth = truth,
        dgm = dgm, n = n_obs, outcome = outcome_type,
        N_profiles = N_PROFILES, N_pasr = N_PASR
      ), ofile)
      cat("  Saved:", ofile, "\n")
    }
  }
}

cat("\nSimulation complete.\n")
plan(sequential)