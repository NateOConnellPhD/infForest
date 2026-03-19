# ============================================================
# infForest worked example
# Run this script to reproduce all README examples
# ============================================================

library(infForest)

# === DGM: 5 predictors ===
set.seed(42)
n <- 400

x1    <- rnorm(n)                               # nonlinear effect (sin)
x2    <- rnorm(n)                               # linear + interaction
trt   <- rbinom(n, 1, 0.4)                      # binary treatment
x4    <- 0.5 * x2 + rnorm(n) * sqrt(0.75)      # correlated with x2
noise <- rnorm(n)                               # null

mu <- 0.8 * sin(1.5 * x1) + 0.4 * x2 + 0.3 * trt + 0.2 * x2 * trt
y_cont <- mu + (0.5 + 0.2 * abs(x1)) * rnorm(n)
y_bin  <- factor(rbinom(n, 1, plogis(mu)), levels = c(0, 1))

dat_cont <- data.frame(x1, x2, trt, x4, noise, y = y_cont)
dat_bin  <- data.frame(x1, x2, trt, x4, noise, y = y_bin)

# DGM truths:
#   x1:    nonlinear, 0.8*sin(1.5*x1)
#   x2:    0.40 + 0.20*mean(trt) ≈ 0.48 per unit (marginal slope Q75 vs Q25)
#   trt:   0.30 + 0.20*mean(x2) ≈ 0.30 (x2 is centered)
#   x4:    0.00 (no direct effect, correlated with x2)
#   noise: 0.00
#   x2:trt interaction: 0.20 (slope of x2 is 0.60 when trt=1, 0.40 when trt=0)

cat("=== DGM Truths ===\n")
cat(sprintf("  trt effect:          %.3f\n", 0.30))
cat(sprintf("  x2 marginal slope:   %.3f\n", 0.40 + 0.20 * mean(trt)))
cat(sprintf("  x4 direct effect:    %.3f\n", 0.00))
cat(sprintf("  noise direct effect: %.3f\n", 0.00))
cat(sprintf("  x2:trt interaction:  %.3f\n", 0.20))
cat("\n")

# ============================================================
# 1. Fit the forest
# ============================================================
fit <- infForest(y ~ ., data = dat_cont, num.trees = 5000,
                 penalize = TRUE, softmax = TRUE)
print(fit)


# ============================================================
# 2. Effect estimation
# ============================================================
cat("\n=== Effect Estimates ===\n")

# Binary: treatment effect
eff_trt <- effect(fit, "trt")
print(eff_trt)

# Continuous: default is Q25 to Q75
eff_x2 <- effect(fit, "x2")
print(eff_x2)

# Multiple quantile comparison points: all pairwise contrasts
eff_x2_multi <- effect(fit, "x2", at = c(0.10, 0.50, 0.90))
print(eff_x2_multi)

# Raw values instead of quantiles
eff_x2_val <- effect(fit, "x2", at = c(-1, 0, 1), type = "value")
print(eff_x2_val)

# Finer grid resolution
eff_x2_fine <- effect(fit, "x2", bw = 10)
print(eff_x2_fine)

# Null: noise (should be near zero)
eff_noise <- effect(fit, "noise")
print(eff_noise)

# Confounder: x4 (correlated with x2, no direct effect)
eff_x4 <- effect(fit, "x4")
print(eff_x4)


# ============================================================
# 3. PASR confidence interval for an effect
# ============================================================
cat("\n=== PASR Effect CI (treatment) ===\n")
pe_trt <- pasr_effect(fit, "trt", verbose = TRUE)
print(pe_trt)

# ============================================================
# 4. Non-Linear Effect curve with PASR confidence bands
# ============================================================

#Without CI
ec <- effect_curve(fit, "x1", bw = 20, q_lo=.02, q_hi=.98)
plot(ec, ylim = c(-1.2, 1.2))
x1_ref <- ec$ref
curve(0.8 * sin(1.5 * x) - 0.8 * sin(1.5 * x1_ref),
      add = TRUE, col = "red", lty = 2, lwd = 2)
legend("topleft", c("infForest", "DGM truth"),
       col = c("black", "red"), lty = c(1, 2), lwd = 2)


#W/ CI
cat("\n=== Effect Curve with PASR Bands ===\n")
x1_grid <- seq(quantile(x1, 0.02), quantile(x1, 0.98), length.out = 12)

curve_ests <- numeric(length(x1_grid))
curve_ses  <- numeric(length(x1_grid))
x1_ref <- median(x1)
for (g in seq_along(x1_grid)) {
  pe <- pasr_effect(fit, "x1", at = c(x1_grid[g], x1_ref), type = "value")
  span <- x1_grid[g] - x1_ref
  # Total contrast = slope * span
  curve_ests[g] <- pe$contrasts$estimate[1] * span
  curve_ses[g]  <- pe$se * abs(span)
  cat(sprintf("  grid[%d] = %.2f: est=%.4f  se=%.4f\n",
              g, x1_grid[g], curve_ests[g], curve_ses[g]))
}


#png("man/figures/effect_curve_x1.png", width = 600, height = 400, res = 100)
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
#dev.off()

# ============================================================
# 5. Interaction estimation
# ============================================================
cat("\n=== Interaction: x2 by trt ===\n")
int_x2trt <- int(fit, "x2", by = "trt")
print(int_x2trt)

# ============================================================
# 6. Covariance floor diagnostic
# ============================================================
cat("\n=== Covariance Floor Diagnostic ===\n")
ct <- ct_diagnose(fit, R = 30L, verbose = TRUE)
print(ct)

#png("man/figures/cov_floor_diagnostic.png", width = 600, height = 400, res = 100)
par(mfrow = c(1, 2))
plot(ct)
plot(ct, by = "x1")
par(mfrow = c(1, 1))
#dev.off()

# ============================================================
# 7. Prediction intervals (continuous)
# ============================================================
cat("\n=== Prediction Intervals (continuous) ===\n")
pi_cont <- pasr_predict(fit, R_max = 50L, verbose = TRUE)
cat("\nFirst 6 observations:\n")
print(head(pi_cont[, c("f_hat", "se", "ci_lower", "ci_upper",
                       "pi_lower", "pi_upper")]))

# Coverage check
y_obs <- dat_cont$y
pi_coverage <- mean(y_obs >= pi_cont$pi_lower & y_obs <= pi_cont$pi_upper)
ci_coverage <- mean(mu >= pi_cont$ci_lower & mu <= pi_cont$ci_upper)
cat(sprintf("\n  Prediction interval coverage: %.1f%%\n", 100 * pi_coverage))
cat(sprintf("  Confidence interval coverage: %.1f%%\n", 100 * ci_coverage))

# Plot
#png("man/figures/pred_int_cont.png", width = 600, height = 400, res = 100)
plot(pi_cont$f_hat, y_obs, pch = 16, cex = 0.5, col = "gray40",
     xlab = "Forest prediction", ylab = "Observed Y",
     main = "Prediction intervals (continuous)")
abline(0, 1, col = "red")
segments(pi_cont$f_hat, pi_cont$pi_lower, pi_cont$f_hat, pi_cont$pi_upper,
         col = rgb(0, 0, 1, 0.08))
#dev.off()
# ============================================================
# 8. Confidence intervals for probabilities (binary)
# ============================================================
cat("\n=== Probability CIs (binary) ===\n")
fit_bin <- infForest(y ~ ., data = dat_bin, num.trees = 3000,
                     penalize = TRUE, softmax = TRUE)
pi_bin <- pasr_predict(fit_bin, R_max = 50L, verbose = TRUE)

cat("\nFirst 6 observations:\n")
print(head(pi_bin[, c("f_hat", "se", "ci_lower", "ci_upper")]))

# Plot sorted probabilities with CI bands
ord <- order(pi_bin$f_hat)
#png("man/figures/pred_int_bin.png", width = 600, height = 400, res = 100)
plot(seq_along(ord), pi_bin$f_hat[ord], type = "l", lwd = 2,
     xlab = "Observation (sorted by P-hat)", ylab = "P(Y = 1 | X)",
     ylim = c(0, 1), main = "Probability estimates with 95% CIs")
polygon(c(seq_along(ord), rev(seq_along(ord))),
        c(pi_bin$ci_lower[ord], rev(pi_bin$ci_upper[ord])),
        col = rgb(0, 0, 1, 0.15), border = NA)
abline(h = 0.5, col = "gray60", lty = 3)
#dev.off()
cat("\n=== Worked example complete ===\n")

# ============================================================
# 10. Predictions at new data points
# ============================================================
cat("\n=== Predictions at new data points ===\n")

set.seed(99)
new_x1    <- rnorm(20)
new_x2    <- rnorm(20)
new_trt   <- rbinom(20, 1, 0.4)
new_x4    <- 0.5 * new_x2 + rnorm(20) * sqrt(0.75)
new_noise <- rnorm(20)
newdata <- data.frame(x1 = new_x1, x2 = new_x2, trt = new_trt,
                      x4 = new_x4, noise = new_noise)

# Continuous: predictions with CIs and PIs
cat("\nContinuous outcome - predictions at new points:\n")
pred_new <- pasr_predict(fit, newdata = newdata, R_max = 50L, verbose = TRUE)
print(pred_new[, c("f_hat", "se", "ci_lower", "ci_upper", "pi_lower", "pi_upper")])

# Binary: predicted probabilities with CIs
cat("\nBinary outcome - predicted probabilities at new points:\n")
pred_new_bin <- pasr_predict(fit_bin, newdata = newdata, R_max = 50L, verbose = TRUE)
print(pred_new_bin[, c("f_hat", "se", "ci_lower", "ci_upper")])

cat("\n=== Worked example complete ===\n")
