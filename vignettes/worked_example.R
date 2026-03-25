# ============================================================
# infForest: Worked Example
# ============================================================
#
# This script demonstrates the full infForest workflow:
#   Part 1 — Pointwise predictions & prediction intervals (ranger + PASR)
#   Part 2 — Effect estimation and inference (infForest)
#   Part 3 — Adjusted means
#   Part 4 — Marginalized predictions
#   Part 5 — Effect transformations (binary outcomes)
#   Part 6 — The $df interface for programmatic access
#   Part 7 — Binary outcomes
#
#
# --- Parallelism ---
#
# infForest has two independent parallelism layers:
#
# 1. R-level parallelism (future/future.apply)
#    PASR fitting runs R independent paired-forest replicates. These are
#    embarrassingly parallel. Loading future and setting plan(multisession)
#    distributes replicates across R worker processes. The number of workers
#    should match your physical cores.
#
#      library(future)
#      library(future.apply)
#      plan(multisession, workers = 8)
#
#    If future is not loaded or plan is sequential, PASR runs sequentially.
#    No code changes needed — the package detects the plan automatically.
#
#    Note: ranger objects with keep.inbag = TRUE can be large (~300 MB for
#    5000 trees). The future framework ships the closure environment to
#    workers, so you may need:
#      options(future.globals.maxSize = 2 * 1024^3)  # 2 GB limit
#
# 2. C++ backend parallelism (OpenMP)
#    The PASR extraction step (scoring cached forests) uses OpenMP threads
#    within a single R process. This is controlled by:
#
#      options(infForest.threads = 8)
#
#    This requires OpenMP support at compile time. On macOS with Apple Clang,
#    you need libomp installed (brew install libomp) and the following in
#    ~/.R/Makevars:
#
#      CPPFLAGS += -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include
#      LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
#
#    On Linux, OpenMP works out of the box with GCC.
#
# Both layers are independent — you can use one, both, or neither.
# ============================================================


library(inf.ranger)
library(infForest)
library(future)
library(future.apply)

plan(multisession, workers = 8)
options(infForest.threads = 8)
options(future.globals.maxSize = 2 * 1024^3)

# Figures directory for README plots
dir.create("man/figures", recursive = TRUE, showWarnings = FALSE)


# ============================================================
# PART 1: Pointwise Predictions (ranger + PASR)
# ============================================================
#
# pasr_predict() works on any fitted ranger model. It computes PASR
# variance estimates and returns prediction intervals (continuous) or
# confidence intervals for P(Y=1) (binary). This is a standalone
# contribution — no infForest object needed.
#
# Workflow:
#   1. Fit ranger on the full dataset
#   2. Call pasr_predict() once — fits nuisance model + paired forests
#   3. Call predict() on the result at any new points (instant)


# --- 1.1 Prediction intervals (continuous) ---

set.seed(42)
n <- 1000
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
y <- 2 * x1 + sin(x2) + rnorm(n)
dat <- data.frame(x1, x2, x3, y)

# keep.inbag = TRUE stores bootstrap membership for the unconditional
# variance estimator (design-point variability, term III).
rf <- ranger(y ~ ., data = dat, num.trees = 5000, keep.inbag = TRUE)

# Fit PASR once. x_conditional = FALSE enables unconditional variance,
# which accounts for both outcome noise and covariate sampling variability.
ps <- pasr_predict(rf, data = dat, x_conditional = FALSE, R = 80, verbose = TRUE)
ps

# Predict at training X — instant, reuses stored forests
pi_train <- predict(ps, newdata = dat[, c("x1","x2","x3")])
head(pi_train[, c("f_hat", "se", "ci_lower", "ci_upper", "pi_lower", "pi_upper")])

# Predict at NEW X — same fitted object, no refitting
set.seed(99)
n_new <- 200
x1_new <- rnorm(n_new); x2_new <- rnorm(n_new); x3_new <- rnorm(n_new)
y_new <- 2 * x1_new + sin(x2_new) + rnorm(n_new)
dat_new <- data.frame(x1 = x1_new, x2 = x2_new, x3 = x3_new)

pi_new <- predict(ps, newdata = dat_new)

# Variance decomposition: unconditional PI variance = Var(f|X) + V_X + sigma2
#   Var(f|X) — conditional forest variance (PASR terms I+II)
#   V_X      — design-point variability (analytic term III)
#   sigma2   — irreducible noise variance
round(c(
  `Var(f|X)` = median(pi_new$var_conditional),
  V_X        = median(pi_new$var_x),
  n_eff      = median(pi_new$n_eff),
  sigma2     = median(pi_new$sigma2_hat)
), 4)

# Coverage at new X (single dataset — illustrative, not a formal assessment)
mean(y_new >= pi_new$pi_lower & y_new <= pi_new$pi_upper)

# Save plot for README
png("man/figures/prediction_intervals_newX.png", width = 700, height = 500)
plot(pi_new$f_hat, y_new, pch = 16, cex = 0.5, col = "gray40",
     xlab = "Forest prediction", ylab = "Observed Y (new data)",
     main = "Prediction intervals (new X)")
abline(0, 1, col = "red")
segments(pi_new$f_hat, pi_new$pi_lower, pi_new$f_hat, pi_new$pi_upper,
         col = rgb(0, 0, 1, 0.15))
dev.off()


# --- 1.2 Probability CIs (binary) ---
#
# For binary outcomes with a probability forest, PASR returns CIs for
# P(Y=1|X). No prediction interval — a new observation is just 0 or 1.

set.seed(43)
n <- 2000
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
x4 <- rnorm(n); x5 <- rbinom(n, 1, 0.4); x6 <- rnorm(n)
x7 <- rnorm(n); noise <- rnorm(n)
p_true <- plogis(1.5*x1 + 0.8*x2 + 0.6*sin(x3) + 0.5*x4 + 0.7*x5 + 0.4*x6 +
                   0.3*x1*x5 + 0.25*x2*x4)
y_bin <- factor(rbinom(n, 1, p_true))
dat_bin <- data.frame(x1, x2, x3, x4, x5, x6, x7, noise, y = y_bin)
rf_bin <- ranger(y ~ ., data = dat_bin, num.trees = 5000, probability = TRUE)

set.seed(100)
n_new <- 300
x1n <- rnorm(n_new); x2n <- rnorm(n_new); x3n <- rnorm(n_new)
x4n <- rnorm(n_new); x5n <- rbinom(n_new, 1, 0.4); x6n <- rnorm(n_new)
x7n <- rnorm(n_new); noisen <- rnorm(n_new)
p_true_new <- plogis(1.5*x1n + 0.8*x2n + 0.6*sin(x3n) + 0.5*x4n + 0.7*x5n + 0.4*x6n +
                       0.3*x1n*x5n + 0.25*x2n*x4n)
dat_bin_new <- data.frame(x1=x1n, x2=x2n, x3=x3n, x4=x4n, x5=x5n, x6=x6n,
                          x7=x7n, noise=noisen)
ps_bin <- pasr_predict(rf_bin, data = dat_bin, R = 80, verbose = TRUE)
pi_bin <- predict(ps_bin, newdata = dat_bin_new)
head(pi_bin[, c("f_hat", "se", "ci_lower", "ci_upper")])

png("man/figures/probability_cis_binary.png", width = 700, height = 500)
plot(pi_bin$f_hat, p_true_new, pch = 16, cex = 0.5, col = "gray40",
     xlab = "Forest prediction P(Y=1)", ylab = "True P(Y=1)",
     main = "Probability CIs (binary)", ylim = c(0, 1), xlim = c(0, 1))
abline(0, 1, col = "red")
segments(pi_bin$f_hat, pi_bin$ci_lower, pi_bin$f_hat, pi_bin$ci_upper,
         col = rgb(0, 0, 1, 0.15))
dev.off()


# --- 1.3 Covariance floor diagnostic ---
#
# ct_diagnose uses the already-fitted PASR object to decompose forest
# variance into MC variance (V/B, vanishes with more trees) and the
# covariance floor (Ct, irreducible structural dependence).

ct <- ct_diagnose(ps, data = dat)
print(ct)


# ============================================================
# PART 2: Effect Estimation and Inference
# ============================================================
#
# infForest provides effect estimates for any predictor type:
#   - Binary:       AIPW-debiased contrast (1 vs 0)
#   - Continuous:   per-unit slope between quantile anchors
#   - Categorical:  all pairwise AIPW contrasts
#
# Workflow:
#   1. Fit infForest (honest cross-fitted forest)
#   2. Call pasr() once — caches paired forests for PASR variance
#   3. Call effect(), summary(), int(), etc. — instant extraction
#
# DGM truth values:
#   trt = 0.30, x2 = 0.40 + 0.20*trt, x4 = 0 (confounder), noise = 0
#   group: B-A = 0.35, C-A = 0.50, C-B = 0.15
#   interaction: x2:trt = 0.20

set.seed(42)
n <- 500
x1 <- rnorm(n); x2 <- rnorm(n); trt <- rbinom(n, 1, 0.4)
x4 <- 0.5 * x2 + rnorm(n) * sqrt(0.75); noise <- rnorm(n)
group <- factor(sample(c("A","B","C"), n, TRUE, c(0.5,0.3,0.2)))
mu <- 0.8*sin(1.5*x1) + 0.4*x2 + 0.3*trt + 0.2*x2*trt +
  ifelse(group=="A", -0.10, ifelse(group=="B", 0.25, 0.40))
y_cont <- mu + (0.5 + 0.2*abs(x1)) * rnorm(n)
y_bin  <- factor(rbinom(n, 1, plogis(mu)), levels = c(0, 1))
dat_cont <- data.frame(x1, x2, trt, x4, noise, group, y = y_cont)
dat_bin  <- data.frame(x1, x2, trt, x4, noise, group, y = y_bin)

# Fit inference forests
fit_cont <- infForest(y ~ ., data = dat_cont, num.trees = 5000,
                      penalize = TRUE, softmax = TRUE)
fit_bin <- infForest(y ~ ., data = dat_bin, num.trees = 5000,
                     penalize = TRUE, softmax = TRUE)

# Run PASR once per model — caches paired forests on the fit object.
# All subsequent effect/predict/summary calls use cached results.
pasr(fit_cont, R = 100)
pasr(fit_bin, R = 100)


# --- 2.1 Binary effects ---

effect(fit_cont, "trt")
effect(fit_cont, "trt", p.value = TRUE)
effect(fit_cont, "trt", p.value = TRUE, marginals = TRUE)

# Compare variance estimators: sandwich (conditional) vs PASR (unconditional)
effect(fit_cont, "trt", variance = "both")

# Binary outcome — same interface
effect(fit_bin, "trt", p.value = TRUE)


# --- 2.2 Continuous effects ---

effect(fit_cont, "x2", bw = 20)
effect(fit_cont, "x2", at = c(0.10, 0.50, 0.90))
effect(fit_cont, "x2", at = c(-1, 0, 1), type = "value")

# Null variables correctly show no effect
effect(fit_cont, "noise", p.value = TRUE)
effect(fit_cont, "x4", p.value = TRUE)


# --- 2.3 Effect curves ---

# Slope curve for x1 — should track 1.2*cos(1.5*x1)
ec_slope <- effect_curve(fit_cont, "x1", q_lo = 0.02, q_hi = 0.98)
png("man/figures/slope_curve_x1.png", width = 700, height = 500)
plot(ec_slope)
midpoints <- (ec_slope$grid[-1] + ec_slope$grid[-length(ec_slope$grid)]) / 2
lines(midpoints, 1.2 * cos(1.5 * midpoints), col = "red", lty = 2, lwd = 2)
legend("topright", c("infForest", "1.2 cos(1.5x)"), col = c("black","red"),
       lty = c(1,2), lwd = 2)
dev.off()

# Level curve for x1 — should track 0.8*sin(1.5*x1)
ec_level <- effect_curve(fit_cont, "x1", q_lo = 0.02, q_hi = 0.98, type = "level")
png("man/figures/level_curve_x1.png", width = 700, height = 500)
plot(ec_level)
x_seq <- seq(min(ec_level$grid), max(ec_level$grid), length = 200)
truth <- 0.8 * sin(1.5 * x_seq)
truth_shifted <- truth - mean(truth) + mean(ec_level$estimate)
lines(x_seq, truth_shifted, col = "red", lty = 2, lwd = 2)
legend("topleft", c("infForest", "0.8 sin(1.5x)"), col = c("black","red"),
       lty = c(1,2), lwd = 2)
dev.off()


# --- 2.4 Categorical effects ---

effect(fit_cont, "group", p.value = TRUE)
effect(fit_cont, "group", at = c("A", "C"))


# --- 2.5 Interactions ---

int(fit_cont, "x2", by = "trt", p.value = TRUE)


# --- 2.6 Summary ---

summary(fit_cont, ~ trt + x2[.10, .90] + noise + group["A", "C"], p.value = TRUE)
summary(fit_bin, ~ trt + x2 + noise, p.value = TRUE)
summary(fit_cont, ~ trt + x2*trt)


# --- 2.7 Variance estimation ---

effect(fit_cont, "trt", variance = "sandwich")
effect(fit_cont, "trt", variance = "pasr")
effect(fit_cont, "trt", variance = "both")
effect(fit_cont, "trt", alpha = 0.10)
effect(fit_cont, "trt", alpha = 0.01)


# ============================================================
# PART 3: Adjusted Means
# ============================================================

forest_means(fit_cont, trt = c(0, 1))
forest_means(fit_cont, x2 = c(-1, 0, 1))
forest_means(fit_cont, trt = c(0, 1), x2 = 0)


# ============================================================
# PART 4: Marginalized Predictions
# ============================================================

predict(fit_cont, newdata = dat_cont[1:5, c("x1","x2","trt","x4","noise","group")])
predict(fit_cont, newdata = data.frame(trt = c(0, 1)))
predict(fit_cont, newdata = data.frame(trt = 1, x2 = c(-1, 0, 1)))

p_curve <- predict(fit_cont, newdata = data.frame(x1 = seq(-2, 2, by = 0.5)))
png("man/figures/marginalized_prediction_x1.png", width = 700, height = 500)
plot(p_curve$x1, p_curve$estimate, type = "l", lwd = 2,
     xlab = "x1", ylab = "E[Y | x1 = x]",
     main = "Marginalized prediction: x1")
polygon(c(p_curve$x1, rev(p_curve$x1)),
        c(p_curve$ci_lower, rev(p_curve$ci_upper)),
        col = rgb(0, 0, 0, 0.12), border = NA)
lines(p_curve$x1, p_curve$estimate, lwd = 2)
dev.off()

predict(fit_cont)  # in-sample


# ============================================================
# PART 5: Effect Transformations (binary outcomes)
# ============================================================

e_bin_trt <- effect(fit_bin, "trt", marginals = TRUE)
transform_effect(e_bin_trt, "OR")
transform_effect(e_bin_trt, "RR")
transform_effect(e_bin_trt, "NNT")


# ============================================================
# PART 6: The $df Interface
# ============================================================

effect(fit_cont, "trt")$df
effect(fit_cont, "trt", marginals = TRUE, p.value = TRUE)$df
effect(fit_cont, "x2")$df
int(fit_cont, "x2", by = "trt")$df
summary(fit_cont, ~ trt + x2 + noise, p.value = TRUE)$df
forest_means(fit_cont, trt = c(0, 1))$df
transform_effect(e_bin_trt, "OR")$df

if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  s_all <- summary(fit_cont, ~ trt + x2 + noise + x4, p.value = TRUE)
  p <- ggplot(s_all$df, aes(x = variable, y = estimate)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2) +
    geom_hline(yintercept = 0, lty = 2) +
    coord_flip() +
    labs(title = "infForest effect estimates", y = "Estimate")
  ggsave("man/figures/forest_plot_effects.png", p, width = 7, height = 5)
}


# ============================================================
# PART 7: Binary Outcomes
# ============================================================

effect(fit_bin, "trt", p.value = TRUE, marginals = TRUE)
effect(fit_bin, "x2")
effect(fit_bin, "noise", p.value = TRUE)
summary(fit_bin, ~ trt + x2 + noise, p.value = TRUE)
predict(fit_bin, newdata = data.frame(trt = c(0, 1)))


plan(sequential)
