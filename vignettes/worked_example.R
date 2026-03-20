# ============================================================
# infForest worked example
# ============================================================

library(infForest)

# === DGM: 5 predictors + 1 categorical ===
set.seed(1000)
n <- 400

x1    <- rnorm(n)                               # nonlinear effect (sin)
x2    <- rnorm(n)                               # linear + interaction
trt   <- rbinom(n, 1, 0.4)                      # binary treatment
x4    <- 0.5 * x2 + rnorm(n) * sqrt(0.75)      # correlated with x2
noise <- rnorm(n)                               # null

# Categorical: 3-level race variable with different effects
cat  <- sample(c("A", "B", "C"), n, replace = TRUE,
                prob = c(0.40, 0.3, 0.1))

cat_effect <- ifelse(cat == "A", -0.10,
                      ifelse(cat == "B",  0.25, 0.40))

mu <- 0.8 * sin(1.5 * x1) + 0.4 * x2 + 0.3 * trt + 0.2 * x2 * trt + cat_effect
y_cont <- mu + (0.5 + 0.2 * abs(x1)) * rnorm(n)
y_bin  <- factor(rbinom(n, 1, plogis(mu)), levels = c(0, 1))

dat_cont <- data.frame(x1, x2, trt, x4, noise, cat,  y = y_cont)
dat_bin  <- data.frame(x1, x2, trt, x4, noise, cat, y = y_bin)

cat("=== DGM Truths ===\n")
cat(sprintf("  trt effect:          %.3f\n", 0.30))
cat(sprintf("  x2 marginal slope:   %.3f\n", 0.40 + 0.20 * mean(trt)))
cat(sprintf("  x4 direct effect:    %.3f\n", 0.00))
cat(sprintf("  noise direct effect: %.3f\n", 0.00))
cat(sprintf("  x2:trt interaction:  %.3f\n", 0.20))
cat(sprintf("  Cat - A:       %.3f\n", 0.35))
cat(sprintf("  Cat - B:       %.3f\n", 0.50))
cat(sprintf("  Cat - C:       %.3f\n", 0.15))
cat(sprintf("  Cat prevalence: A=%.0f%%, B=%.0f%%, C=%.0f%%\n",
            100*mean(race=="A"), 100*mean(race=="B"), 100*mean(race=="C")))
cat("\n")



# ============================================================
# 1. Fit the forest
# ============================================================
fit <- infForest(y ~ ., data = dat_cont, num.trees = 5000,
                 penalize = TRUE, softmax = TRUE)
print(fit)


# ============================================================
# 2. Effect estimation — point estimates with sandwich CIs
#    Default: variance = "sandwich", ci = TRUE
# ============================================================
cat("\n=== Effect Estimates (with sandwich CIs) ===\n")

# Binary: treatment effect
effect(fit, "trt")

# Continuous: default Q25 to Q75 slope
effect(fit, "x2")

# Multiple comparison points
effect(fit, "x2", at = c(0.10, 0.50, 0.90))

# Raw values instead of quantiles
effect(fit, "x2", at = c(-1, 0, 1), type = "value")

# Finer grid resolution
effect(fit, "x2", bw = 10)

# Null: should be near zero, CI should contain zero
effect(fit, "noise")

# Confounder: x4 correlated with x2 but no direct effect
effect(fit, "x4")


# ============================================================
# 3. Categorical effects
# ============================================================
cat("\n=== Categorical (all pairwise contrasts) ===\n")

# All pairwise contrasts (default)
effect(fit, "cat")

# Specific contrast: A vs B
effect(fit, "cat", at = c("A", "B"), variance="both")

# Specific contrast: A vs C
effect(fit, "cat", at = c("A", "C"), variance="both")

# B vs C (small group contrast — expect wider CI)
effect(fit, "cat", at = c("B", "C"))


# ============================================================
# 4. Point estimates only (no CI overhead)
# ============================================================
cat("\n=== Point estimates only ===\n")

effect(fit, "trt", ci = FALSE)
effect(fit, "x2", ci = FALSE)
effect(fit, "race", ci = FALSE)


# ============================================================
# 5. PASR variance (more thorough, slower)
# ============================================================
cat("\n=== PASR variance for treatment effect ===\n")

effect(fit, "trt", variance = "pasr", verbose = TRUE)

# Both sandwich and PASR — reports diagnostic ratio rho_V
effect(fit, "trt", variance = "both", verbose = TRUE)


# ============================================================
# 6. Nonlinear effect curve with PASR confidence bands
# ============================================================
cat("\n=== Effect curve for x1 ===\n")

# Quick curve (no CI)
ec <- effect_curve(fit, "x1", bw = 20, q_lo = 0.02, q_hi = 0.98)
plot(ec, ylim = c(-1.2, 1.2))
x1_ref <- ec$ref
curve(0.8 * sin(1.5 * x) - 0.8 * sin(1.5 * x1_ref),
      add = TRUE, col = "red", lty = 2, lwd = 2)
legend("topleft", c("infForest", "DGM truth"),
       col = c("black", "red"), lty = c(1, 2), lwd = 2)

# Curve with PASR bands at 12 grid points
x1_grid <- seq(quantile(x1, 0.02), quantile(x1, 0.98), length.out = 12)
x1_ref <- median(x1)

curve_ests <- curve_ses <- numeric(length(x1_grid))
for (g in seq_along(x1_grid)) {
  pe <- effect(fit, "x1", at = c(x1_grid[g], x1_ref), type = "value",
               variance = "pasr")
  span <- x1_grid[g] - x1_ref
  curve_ests[g] <- pe$contrasts$estimate[1] * span
  curve_ses[g]  <- pe$se * abs(span)
}

plot(x1_grid, curve_ests, type = "l", lwd = 2,
     xlab = "x1", ylab = "Effect relative to median",
     main = "Effect curve for x1 with 95% PASR bands",
     ylim = range(c(curve_ests - 1.96 * curve_ses,
                    curve_ests + 1.96 * curve_ses)))
polygon(c(x1_grid, rev(x1_grid)),
        c(curve_ests - 1.96 * curve_ses, rev(curve_ests + 1.96 * curve_ses)),
        col = rgb(0, 0, 0, 0.15), border = NA)
curve(0.8 * sin(1.5 * x) - 0.8 * sin(1.5 * x1_ref),
      add = TRUE, col = "red", lty = 2, lwd = 2)
abline(h = 0, col = "gray60", lty = 3)
legend("topleft", c("infForest ± 95% CI", "DGM truth"),
       col = c("black", "red"), lty = c(1, 2), lwd = 2)


# ============================================================
# 7. Interaction estimation — with CIs
# ============================================================
cat("\n=== Interaction: x2 by trt ===\n")

# Sandwich CI (default)
int(fit, "x2", by = "trt")

# PASR CI
int(fit, "x2", by = "trt", variance = "pasr", verbose = TRUE)


# ============================================================
# 8. Summary — multiple effects at once
# ============================================================
cat("\n=== Summary ===\n")

summary(fit, ~ trt + x2[.5, .8] + x4[.1,.9] + noise + race)

# Specific race contrast in summary
summary(fit, ~ trt + race["white", "black"])


# ============================================================
# 9. Covariance floor diagnostic
# ============================================================
cat("\n=== Covariance Floor Diagnostic ===\n")

ct <- ct_diagnose(fit, R = 30L, verbose = TRUE)
print(ct)
par(mfrow = c(1, 2))
plot(ct)
plot(ct, by = "x1")
par(mfrow = c(1, 1))


# ============================================================
# 10. Prediction intervals (continuous)
# ============================================================
cat("\n=== Prediction Intervals (continuous) ===\n")

pi_cont <- pasr_predict(fit, R_max = 50L, verbose = TRUE)
head(pi_cont[, c("f_hat", "se", "ci_lower", "ci_upper",
                 "pi_lower", "pi_upper")])

# Coverage check
y_obs <- dat_cont$y
pi_cov <- mean(y_obs >= pi_cont$pi_lower & y_obs <= pi_cont$pi_upper)
ci_cov <- mean(mu >= pi_cont$ci_lower & mu <= pi_cont$ci_upper)
cat(sprintf("  PI coverage: %.1f%%\n", 100 * pi_cov))
cat(sprintf("  CI coverage: %.1f%%\n", 100 * ci_cov))

plot(pi_cont$f_hat, y_obs, pch = 16, cex = 0.5, col = "gray40",
     xlab = "Forest prediction", ylab = "Observed Y",
     main = "Prediction intervals (continuous)")
abline(0, 1, col = "red")
segments(pi_cont$f_hat, pi_cont$pi_lower, pi_cont$f_hat, pi_cont$pi_upper,
         col = rgb(0, 0, 1, 0.08))


# ============================================================
# 11. Probability CIs (binary)
# ============================================================
cat("\n=== Probability CIs (binary) ===\n")

fit_bin <- infForest(y ~ ., data = dat_bin, num.trees = 3000,
                     penalize = TRUE, softmax = TRUE)
pi_bin <- pasr_predict(fit_bin, R_max = 50L, verbose = TRUE)
head(pi_bin[, c("f_hat", "se", "ci_lower", "ci_upper")])

ord <- order(pi_bin$f_hat)
plot(seq_along(ord), pi_bin$f_hat[ord], type = "l", lwd = 2,
     xlab = "Observation (sorted by P-hat)", ylab = "P(Y = 1 | X)",
     ylim = c(0, 1), main = "Probability estimates with 95% CIs")
polygon(c(seq_along(ord), rev(seq_along(ord))),
        c(pi_bin$ci_lower[ord], rev(pi_bin$ci_upper[ord])),
        col = rgb(0, 0, 1, 0.15), border = NA)
abline(h = 0.5, col = "gray60", lty = 3)


# ============================================================
# 12. Predictions at new data
# ============================================================
cat("\n=== Predictions at new data ===\n")

set.seed(99)
newdata <- data.frame(
  x1 = rnorm(20), x2 = rnorm(20), trt = rbinom(20, 1, 0.4),
  x4 = rnorm(20), noise = rnorm(20),
  race = factor(sample(c("white","black","asian"), 20, replace = TRUE),
                levels = c("white","black","asian"))
)

# Continuous: predictions with CIs and PIs
pasr_predict(fit, newdata = newdata, R_max = 50L, verbose = TRUE)

# Binary: predicted probabilities with CIs
pasr_predict(fit_bin, newdata = newdata, R_max = 50L, verbose = TRUE)

cat("\n=== Worked example complete ===\n")
