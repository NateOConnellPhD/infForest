# infForest

> Nonparametric inference from random forests — effects, predictions, and uncertainty quantification

**infForest** is an R package that turns random forests into full inferential procedures. It provides effect estimates for any predictor, nonlinear effect curves, interaction detection, prediction intervals, and confidence intervals for predicted probabilities — all without functional form assumptions.

## Quick start

```r
library(infForest)

# Fit (continuous or binary outcome auto-detected)
fit <- infForest(y ~ ., data = dat, num.trees = 3000, penalize = TRUE, softmax = TRUE)

# Effect estimate (any predictor — binary or continuous)
effect(fit, "treatment")

# Instant confidence interval (sandwich SE, ~seconds)
pasr_effect(fit, "treatment", variance_method = "sandwich")

# Thorough confidence interval (PASR, ~minutes)
pasr_effect(fit, "treatment", variance_method = "pasr", verbose = TRUE)

# Nonlinear effect curve with confidence bands + omnibus test
pc <- pasr_curve(fit, "x1", verbose = TRUE)
plot(pc)

# Prediction intervals at new data points
pasr_predict(fit, newdata = new_obs, verbose = TRUE)
```

The typical workflow is: **fit → effect → CI → curve → predict**. Start with `effect()` for point estimates, add `pasr_effect(variance_method = "sandwich")` for instant SEs, upgrade to `pasr_effect()` (default `variance_method = "both"`) when you need thorough inference, and use `pasr_curve()` for nonlinear curves with bands.

## Installation

```r
# Requires the inf.ranger fork of ranger
devtools::install_github("NateOConnellPhD/inf.ranger")
devtools::install_github("NateOConnellPhD/infForest")
```

### About inf.ranger

infForest uses [ranger](https://github.com/imbs-hl/ranger) as its forest engine. The `inf.ranger` package is a minimal fork of ranger with two targeted modifications to the split selection code — the tree-growing algorithm, prediction machinery, and all other ranger internals are unchanged.

**`penalize.split.competition`** — At each node, CART selects the variable with the largest impurity reduction. But continuous variables evaluate many more candidate split points than binary variables, giving them a structural advantage: the maximum of *M* noisy candidates is systematically larger than the maximum of 1, even under the null. This is the search advantage (a winner's curse proportional to $\sqrt{2 \log M}$). The standardized criterion subtracts the expected search advantage from each variable's best split score, producing a level comparison across variable types. The correction is closed-form and adds negligible computation.

**`softmax.split`** — Standard CART selects the single best variable at each node (argmax). Softmax replaces this with probabilistic selection: variables are chosen with probability proportional to $\exp(\tau \cdot \tilde{G}_j)$, where $\tilde{G}_j$ is the penalized criterion. This increases the inclusion rate for variables with moderate but real signal that would otherwise be crowded out by stronger predictors at every node. $\tau$ is set automatically. Requires `penalize.split.competition = TRUE`.

Both modifications operate only at the moment of variable selection within each node. Everything downstream — split point selection, daughter node assignment, leaf predictions, OOB estimation, prediction — is identical to standard ranger.

## Important: data requirements

Like any random forest, **infForest requires complete data with no missing values**. If your data has missing values, you must handle them before fitting:

- **Complete cases:** `dat <- dat[complete.cases(dat), ]` — simple but discards rows.
- **Imputation:** Use `mice`, `missForest`, `Amelia`, or similar. Multiple imputation with pooled inference is preferred when missingness is substantial.

**Supported variable types:** Numeric (continuous) and two-level factors (binary) for the outcome. Predictors can be numeric or integer. Multi-level factors are not currently supported as predictors — convert to dummy variables first.

---

## Worked example

A complete walkthrough with a simple DGM. The runnable script is in [`vignettes/worked_example.R`](vignettes/worked_example.R).

### 1. Data-generating mechanism

We simulate five predictors with known relationships to the outcome: a nonlinear continuous effect (sin curve), a linear continuous effect that interacts with treatment, a binary treatment, a correlated confounder with no direct effect, and pure noise. We create both a continuous and binary version of the outcome for demonstrating both outcome types.

```r
library(infForest)
set.seed(42)
n <- 400

x1    <- rnorm(n)                               # nonlinear (sin)
x2    <- rnorm(n)                               # linear + interaction
trt   <- rbinom(n, 1, 0.4)                      # binary treatment
x4    <- 0.5 * x2 + rnorm(n) * sqrt(0.75)      # correlated with x2
noise <- rnorm(n)                               # null

mu <- 0.8 * sin(1.5 * x1) + 0.4 * x2 + 0.3 * trt + 0.2 * x2 * trt
y_cont <- mu + (0.5 + 0.2 * abs(x1)) * rnorm(n)    # heteroscedastic noise
y_bin  <- factor(rbinom(n, 1, plogis(mu)), levels = c(0, 1))

dat_cont <- data.frame(x1, x2, trt, x4, noise, y = y_cont)
dat_bin  <- data.frame(x1, x2, trt, x4, noise, y = y_bin)
```

True effects: trt = 0.30, x2 marginal slope ≈ 0.48 (averaged over treatment groups, since trt interacts with x2), x4 = 0 (correlated with x2 but no direct effect), noise = 0, and x2 × trt interaction = 0.20 per unit. The binary outcome uses the same conditional mean through a logistic link.

### 2. Fitting the forest

`infForest()` fits a cross-fitted honest random forest. The data is repeatedly split into build and estimation folds: trees are grown on the build fold, and all downstream estimates (effects, predictions, scores) use only estimation-fold outcomes. This eliminates adaptive bias. Multiple independent splits (`honesty.splits`) are averaged to reduce fold-assignment variance.

The outcome type is auto-detected: numeric → continuous (regression), two-level factor → binary (probability forest).

```r
# Continuous outcome
fit <- infForest(y ~ ., data = dat_cont, num.trees = 3000,
                 penalize = TRUE, softmax = TRUE)
fit
#> Inference Forest
#>   Outcome type:     continuous
#>   Observations:     400
#>   Predictors:       5
#>   Trees per forest: 3000
#>   mtry:             2
#>   min.node.size:    10
#>   Honest:           yes
#>   Honesty splits:   5
#>   Penalized splits: TRUE
#>   Softmax splits:   TRUE

# Binary outcome (used later for probability predictions)
fit_bin <- infForest(y ~ ., data = dat_bin, num.trees = 3000,
                     penalize = TRUE, softmax = TRUE)
```

Fitting takes a few seconds for n = 400 with 3000 trees. For larger datasets (n > 5000), expect 30–120 seconds depending on the number of predictors and trees.

**Key parameters for `infForest()`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num.trees` | 5000 | Trees per forest. More trees reduce Monte Carlo variance. 3000+ recommended for inference. |
| `mtry` | $\lfloor\sqrt{p}\rfloor$ | Number of candidate variables evaluated at each split. |
| `min.node.size` | 10 | Minimum observations in a terminal node. Controls resolution: smaller values allow finer conditioning but increase variance per leaf. |
| `honesty.splits` | 5 | Number of independent build/estimation fold assignments to average. Reduces fold-assignment variance. |
| `penalize` | TRUE | Use the standardized splitting criterion. Corrects variable selection bias from the search advantage. Always use for inference. |
| `softmax` | FALSE | Probabilistic variable selection via softmax. Set TRUE when you have weak signals (e.g., rare binary predictors alongside strong continuous ones). Requires `penalize = TRUE`. |
| `sample.fraction` | 1.0 | Fraction of fold observations to sample per tree. Default uses all build-fold observations. |
| `replace` | FALSE | Sample with replacement. FALSE maximizes effective sample size per tree. |
| `seed` | NULL | Random seed for reproducibility. |

### 3. Effect estimation

`effect()` estimates the population-averaged effect of any predictor using AIPW (Augmented Inverse-Propensity Weighted) scores with honest cross-fitting. The estimator combines forest prediction contrasts with propensity-weighted residuals, achieving double robustness: consistent if either the forest predictions or the propensity model is consistent.

**What the estimate means:**

- **Binary predictors:** The estimate is the average difference in the conditional mean when the predictor is 1 vs 0, adjusting for all other covariates: $E[f(X^{(j=1)}) - f(X^{(j=0)})]$. Under exchangeability, this is the average contrasted effect.
- **Continuous predictors:** The estimate is the average per-unit slope of the effect curve between two comparison points. For example, "Q25 to Q75: 0.50" means that on average, a one-unit increase in the predictor between the 25th and 75th percentile is associated with a 0.50 increase in the outcome. This is computed from the integrated AIPW effect curve, so it captures nonlinearity — the slope between different comparison points can differ.

```r
# Binary predictor: population-averaged treatment effect
# True value: 0.30
effect(fit, "trt")
#>   Variable:    trt
#>   Type:        binary
#>   Estimate:    0.2862

# Continuous predictor: per-unit slope from Q25 to Q75 (default)
# True x2 slope ≈ 0.48 (averaging 0.40 at trt=0 and 0.60 at trt=1)
effect(fit, "x2")
#>   Pairwise contrasts (per unit):
#>     Q25 to Q75  [-0.701, 0.600]:  0.8875

# Multiple quantile comparison points — returns all pairwise contrasts.
# Different slopes across intervals reveal nonlinearity.
effect(fit, "x2", at = c(0.10, 0.50, 0.90))
#>   Pairwise contrasts (per unit):
#>     Q10 to Q50  [-1.383, -0.068]:  0.8810
#>     Q10 to Q90  [-1.383, 1.182]:   0.8332
#>     Q50 to Q90  [-0.068, 1.182]:   0.7830

# Use raw covariate values instead of quantiles
effect(fit, "x2", at = c(-1, 0, 1), type = "value")
#>   Pairwise contrasts (per unit):
#>     -1 to 0  [-1.000, 0.000]:  0.8299
#>     -1 to 1  [-1.000, 1.000]:  0.7602
#>      0 to 1  [0.000, 1.000]:   0.6905

# Null predictor: should be near zero. True value: 0.
effect(fit, "noise")
#>     Q25 to Q75  [-0.669, 0.713]:  0.1507

# Confounder: correlated with x2 but no direct effect.
# AIPW adjusts for the confounding — estimate should be near zero.
# True value: 0.
effect(fit, "x4")
#>     Q25 to Q75  [-0.765, 0.585]:  -0.0517
```

`effect()` runs in under a second. It returns point estimates only — no standard errors. For inference, see Section 4.

**Key parameters for `effect()`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `at` | `c(0.25, 0.75)` | Comparison points for continuous predictors. Two points → one contrast; more → all pairwise contrasts. |
| `type` | `"quantile"` | How to interpret `at`. `"quantile"` treats values as quantiles of the predictor's distribution; `"value"` treats them as raw covariate values. |
| `bw` | 20 | Bandwidth for the internal effect curve grid. Controls honest observations per grid interval. Smaller = finer resolution but noisier. |
| `q_lo`, `q_hi` | 0.10, 0.90 | Quantile bounds for the internal grid. The curve is estimated over this range. |
| `subset` | NULL | Integer indices for conditional effects within a subgroup (e.g., `subset = which(dat$age > 65)`). |

### 4. Variance estimation and confidence intervals

`pasr_effect()` adds standard errors and confidence intervals to any effect estimate. It supports three variance estimation methods via the `variance_method` parameter. The choice controls a speed–thoroughness tradeoff.

#### The sandwich (influence function) estimator

The sandwich estimator computes the sample variance of the per-observation AIPW influence function scores. It is computationally free — it uses scores already produced during effect estimation, requiring no additional forests. This is the nonparametric analogue of the Huber-White robust standard error.

The sandwich SE targets the finite-sample variance of the deployed estimator. As both the forest and propensity model improve, it converges to the semiparametric efficiency bound.

**When to use:** Exploratory analysis, screening many variables, or any time you want instant CIs. Good default for large samples.

#### PASR (Procedure-Aligned Synthetic Resampling)

PASR estimates the covariance floor — the irreducible variance that persists even with infinite trees, arising from structural dependence between trees sharing training data. It generates synthetic outcomes from a fitted nuisance model, refits paired forests on each synthetic dataset with shared fold assignments, and estimates the covariance floor from the cross-covariance of paired estimates.

PASR is more computationally expensive (fits R × 4 ranger forests, typically 80–800 fits), but it captures variance sources that the sandwich may understate, particularly in small samples where the structural dependence between trees is large relative to the estimation variance.

**When to use:** Final inference for a paper or report. When you need the most reliable CI. When n is small and you want to be conservative.

#### Using `pasr_effect()`

```r
# Sandwich only — instant, no paired forests (~1-2 seconds)
pasr_effect(fit, "trt", variance_method = "sandwich")
#>   Estimate:    0.298
#>   SE (sandwich): 0.0727
#>   95% CI:      [0.1555, 0.4405]
#>   p-value:     3.98e-05

# PASR only — fits paired forests (~1-3 minutes at n=400)
pasr_effect(fit, "trt", variance_method = "pasr", verbose = TRUE)
#>   PASR R=20: C_psi=0.006248  rel_change=Inf  stable=0/2
#>   PASR R=30: C_psi=0.006104  rel_change=0.0231  stable=0/2
#>   PASR R=40: C_psi=0.005976  rel_change=0.0208  stable=1/2
#>   PASR converged at R=40 (C_psi=0.005976)
#>   Estimate:    0.298
#>   SE (pasr):   0.0904
#>   95% CI:      [0.1207, 0.4752]
#>   p-value:     0.000983

# Both (default) — computes both and reports the diagnostic ratio rho_V
pe_trt <- pasr_effect(fit, "trt", verbose = TRUE)
pe_trt
#>   Estimate:    0.298
#>   SE (both):   0.0904
#>   95% CI:      [0.1207, 0.4752]
#>   p-value:     0.000983
#>   SE (sand):   0.0727  |  95% CI: [0.1555, 0.4405]
#>   rho_V:       0.65
```

**Interpreting `rho_V`:** The ratio of sandwich variance to PASR variance. Values near 1 mean the two estimators agree. Values well below 1 (like 0.65 above) indicate the sandwich is underestimating variance — the covariance floor is contributing meaningfully and only PASR captures it. This is typical in moderate-sample settings. Values above 1 may indicate PASR overestimation (rare, usually from noisy nuisance model).

**Saving time with pre-computed nuisance models:** If you are calling `pasr_effect()` for multiple variables, you can estimate the nuisance model once and reuse it:

```r
nuis <- estimate_nuisance(fit)
pe_trt  <- pasr_effect(fit, "trt", nuisance = nuis, verbose = TRUE)
pe_x2   <- pasr_effect(fit, "x2", nuisance = nuis, verbose = TRUE)
pe_x4   <- pasr_effect(fit, "x4", nuisance = nuis, verbose = TRUE)
```

**Key parameters for `pasr_effect()`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `variance_method` | `"both"` | `"sandwich"` for instant SE, `"pasr"` for PASR only, `"both"` for both plus `rho_V`. |
| `R_min`, `R_max` | 20, 200 | Minimum and maximum PASR replicates. More replicates → more stable C_psi estimate. |
| `batch_size` | 10 | Replicates between convergence checks. |
| `tol` | 0.05 | Relative change threshold for convergence. |
| `n_stable` | 2 | Consecutive stable batches required to stop early. |
| `B_mc` | 500 | Trees per paired forest in PASR. Can be smaller than the deployed forest — PASR estimates variance, not the effect. |
| `alpha` | 0.05 | Significance level for confidence intervals. |
| `nuisance` | NULL | Pre-computed nuisance model from `estimate_nuisance()`. If NULL, estimated automatically. |

### 5. Nonlinear effect curves

**`effect_curve()`** traces the relationship $E[Y \mid X_j = x] - E[Y \mid X_j = \text{ref}]$ nonparametrically across the range of a continuous predictor. No functional form is assumed. The curve is estimated by integrating AIPW slopes over a fine grid of the predictor's values. The result is centered at a reference value (default: the median) so that the curve shows how the outcome changes relative to a typical observation.

```r
ec <- effect_curve(fit, "x1")
plot(ec)
```

`effect_curve()` returns the point estimate curve only, with no uncertainty quantification. For confidence bands, use `pasr_curve()`.

**`pasr_curve()`** adds confidence bands and an omnibus test by fitting paired forests once per PASR replicate and extracting the full curve from each. This is dramatically faster than calling `pasr_effect()` in a loop over grid points: R × 4 ranger fits total, regardless of how many grid points.

```r
pc <- pasr_curve(fit, "x1", verbose = TRUE)
#>   PASR R=20: median_C=0.001234  rel_change=Inf  stable=0/2
#>   ...
#>   PASR converged at R=40

pc
#> Inference Forest Effect Curve
#>   Variable:    x1
#>   Grid range:  -1.246 to 1.383
#>   Reference:   -0.025
#>   Intervals:   10
#>   Curve range: -0.6821 to 0.5432
#>   Max SE:      0.2134
#>   Variance:    both
#>   Omnibus:     chi2 = 42.15, df = 10, p = 1.19e-05

# Plot with bands and true DGM overlaid
plot(pc)
x1_ref <- median(x1)
curve(0.8 * sin(1.5 * x) - 0.8 * sin(1.5 * x1_ref),
      add = TRUE, col = "red", lty = 2, lwd = 2)
legend("topleft", c("infForest ± 95% CI", "DGM truth"),
       col = c("black", "red"), lty = c(1, 2), lwd = 2)
```

![Effect curve for x1 with PASR confidence bands](man/figures/effect_curve_x1.png)

**The omnibus test** (`pc$omnibus_pval`) provides a single p-value for the global null $H_0: f(x_1) = \text{constant}$. It tests whether the predictor has *any* association (linear or nonlinear) with the outcome, using the joint covariance matrix across all grid points. This is a chi-squared test with degrees of freedom equal to the number of grid points minus the reference.

**Key parameters for `pasr_curve()`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q_lo`, `q_hi` | 0.10, 0.90 | Quantile bounds for the curve grid. |
| `bw` | 20 | Bandwidth (honest obs per grid interval). |
| `ref` | `median(x)` | Reference value where the curve equals zero. |
| `variance_method` | `"both"` | Same as `pasr_effect()`: `"sandwich"`, `"pasr"`, or `"both"`. |
| PASR params | — | `R_min`, `R_max`, `batch_size`, `tol`, `n_stable`, `B_mc`, `alpha` — same as `pasr_effect()`. |

### 6. Interaction estimation

`int()` estimates the effect of one predictor within subgroups defined by another predictor. For binary conditioning variables, it gives the effect of the focal variable within each group and the difference between groups (the interaction contrast). For continuous conditioning variables, it estimates the effect within user-defined quantile bands.

```r
# x2 effect within treatment groups
# True: slope = 0.40 when trt=0, slope = 0.60 when trt=1, difference = 0.20
int(fit, "x2", by = "trt")
#> Inference Forest Interaction
#>   Variable:   x2
#>   By:         trt (binary)
#>
#>   Subgroup effects:
#>     trt = 1                           0.6062  (per unit)
#>     trt = 0                           0.7645  (per unit)
#>
#>   Pairwise differences:
#>     trt = 1 vs trt = 0               -0.1583
```

The "pairwise difference" is the effect in group 1 minus the effect in group 0. A negative value here means x2 has a *weaker* per-unit slope in the treatment group than in the control group. The true interaction is 0.20 (treatment group has a steeper slope), so the forest estimate is in the right direction but noisy at n = 400 — this is expected for interaction estimates, which require substantially more data than main effects.

```r
# Three-way: x2 effect by trt, conditional on x1 > 0
int(fit, "x2", by = "trt", subset = which(dat_cont$x1 > 0))

# Continuous conditioning: x1 effect in low vs high x2 bands
int(fit, "x1", by = "x2",
    by_at = list(c(0.10, 0.25), c(0.75, 0.90)))
```

**Key parameters for `int()`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `by` | *(required)* | Name of the conditioning variable. |
| `by_at` | NULL | For continuous conditioning variables: a list of quantile pairs `c(lo, hi)` defining subgroup bands. |
| `subset` | NULL | Integer indices for higher-order conditioning. |

### 7. Batch summary

`summary()` evaluates multiple effects and interactions in one call using a formula interface. Useful for screening all predictors.

```r
# All main effects
summary(fit, ~ x1 + x2 + trt + x4 + noise)

# Main effects plus an interaction
summary(fit, ~ x1 + x2 * trt)

# Custom comparison points for specific variables
summary(fit, ~ x1[0.10, 0.50, 0.90] + x2)
```

### 8. Prediction intervals and probability confidence intervals

`pasr_predict()` constructs uncertainty intervals for individual predictions. The type of interval depends on the outcome:

- **Continuous outcomes → prediction intervals (PIs):** Cover where a *new observation* Y would fall, accounting for both estimation uncertainty and residual variance $\sigma^2(x)$. The output columns are `pi_lower` and `pi_upper`.
- **Binary outcomes → confidence intervals (CIs):** Cover the true conditional probability $P(Y = 1 \mid X = x)$. The output columns are `ci_lower` and `ci_upper`.

`pasr_predict()` auto-detects the outcome type and returns the appropriate interval.

#### Continuous outcome — prediction intervals

```r
pi_cont <- pasr_predict(fit, R_max = 50L, verbose = TRUE)
#>   PASR R=20: median_Ct=0.106982  ...
#>   ...

head(pi_cont[, c("f_hat", "se", "pi_lower", "pi_upper")])
#>      f_hat     se  pi_lower  pi_upper
#> 1  1.4204 0.9328  -0.4078    3.2486
#> 2 -0.5748 0.5246  -1.6029    0.4534
#> ...

# Plot: observed vs predicted with prediction intervals
plot(pi_cont$f_hat, dat_cont$y, pch = 16, cex = 0.5, col = "gray40",
     xlab = "Forest prediction", ylab = "Observed Y",
     main = "Prediction intervals (continuous)")
abline(0, 1, col = "red")
segments(pi_cont$f_hat, pi_cont$pi_lower, pi_cont$f_hat, pi_cont$pi_upper,
         col = rgb(0, 0, 1, 0.08))
```

![Prediction intervals for continuous outcome](man/figures/pred_int_cont.png)

#### Binary outcome — probability confidence intervals

```r
pi_bin <- pasr_predict(fit_bin, R_max = 50L, verbose = TRUE)

# Sorted probability plot with CI bands
ord <- order(pi_bin$f_hat)
plot(seq_along(ord), pi_bin$f_hat[ord], type = "l", lwd = 2,
     xlab = "Observation (sorted by P-hat)", ylab = "P(Y = 1 | X)",
     ylim = c(0, 1), main = "Probability estimates with 95% CIs")
polygon(c(seq_along(ord), rev(seq_along(ord))),
        c(pi_bin$ci_lower[ord], rev(pi_bin$ci_upper[ord])),
        col = rgb(0, 0, 1, 0.15), border = NA)
abline(h = 0.5, col = "gray60", lty = 3)
```

![Confidence intervals for predicted probabilities](man/figures/pred_int_bin.png)

#### Predictions at new data points

A fitted forest generates predictions with uncertainty estimates at new covariate values — the standard use case of fitting once and predicting at new observations.

```r
set.seed(99)
newdata <- data.frame(
  x1    = rnorm(20),
  x2    = rnorm(20),
  trt   = rbinom(20, 1, 0.4),
  x4    = 0.5 * rnorm(20) + rnorm(20) * sqrt(0.75),
  noise = rnorm(20)
)

# Continuous: predictions with PIs
pred_new <- pasr_predict(fit, newdata = newdata, R_max = 50L, verbose = TRUE)
head(pred_new[, c("f_hat", "se", "pi_lower", "pi_upper")])

# Binary: predicted probabilities with CIs
pred_new_bin <- pasr_predict(fit_bin, newdata = newdata, R_max = 50L, verbose = TRUE)
head(pred_new_bin[, c("f_hat", "se", "ci_lower", "ci_upper")])
```

**Key parameters for `pasr_predict()`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `newdata` | NULL | Data frame of new observations. Column names must match training data. If NULL, predicts at training points. |
| `R_min`, `R_max` | 20, 200 | PASR replicate bounds. |
| `B_mc` | 500 | Trees per paired forest in PASR. |
| `alpha` | 0.05 | Significance level for intervals. |

### 9. Covariance floor diagnostic

`ct_diagnose()` estimates the covariance floor $C_T(x)$ at every observation — the irreducible prediction variance that persists even with infinitely many trees. This diagnostic reveals where the forest is structurally uncertain: typically in sparse regions, near the boundary of the predictor space, or where the signal is complex.

```r
ct <- ct_diagnose(fit, R = 30L, verbose = TRUE)
ct
#> Covariance Floor Diagnostic (C_T)
#>   Prediction points: 400
#>   C_T summary:
#>     Mean:   0.204829
#>     Median: 0.099561
#>     ...

# C_T vs fitted value, C_T vs a covariate
par(mfrow = c(1, 2))
plot(ct)
plot(ct, by = "x1")
par(mfrow = c(1, 1))
```

![Covariance floor diagnostic](man/figures/cov_floor_diagnostic.png)

---

## How it works

### Honest estimation

Data is split into build and estimation folds. Trees are grown on the build fold; all effect estimates use only estimation-fold outcomes. This eliminates adaptive bias — the tree structure is determined independently of the outcome values used for estimation. Multiple independent splits (`honesty.splits`) are averaged to reduce fold-assignment variance.

### AIPW debiasing

Forest prediction contrasts are biased because trees that split on a predictor absorb its signal into the tree structure. The AIPW correction adds propensity-weighted honest residuals. Concretely, the per-observation score for the effect of $X_j$ is:

$$\phi_k = [\hat{f}(X_k^{(j=a)}) - \hat{f}(X_k^{(j=b)})] + \hat\omega_k \cdot [Y_k - \hat{f}(X_k)]$$

The first term is the forest's prediction contrast — what the forest thinks would happen if $X_j$ were set to $a$ vs $b$, holding everything else constant. The second term corrects for bias: $\hat\omega_k$ is a propensity-based weight (how much observation $k$'s residual is informative about the effect of $X_j$), and $Y_k - \hat{f}(X_k)$ is the honest residual. The effect estimate is the average of these scores: $\hat\psi = \bar\phi$.

This construction makes the bias proportional to the *product* of the prediction error and the propensity error — small even when both models are moderately wrong (double robustness).

### Variance estimation

Two variance estimators are available:

**Sandwich (influence function):** The sample variance of the AIPW scores, $\hat{V}_{\text{IF}} = \frac{1}{n(n-1)} \sum_k (\phi_k - \bar\phi)^2$. Computationally free. Asymptotically exact.

**PASR:** Estimates the covariance floor $C_\Psi$ — the variance component from structural tree dependence — by refitting paired forests on synthetic data and computing cross-covariance: $\hat{C}_\Psi = \text{cov}(\hat\psi^A, \hat\psi^B)$ across R synthetic replicates. The total PASR variance adds the Monte Carlo component (estimated from honesty-split variation).

### Standardized splitting and softmax selection

The standardized criterion corrects variable selection bias from the search advantage; softmax selection ensures moderate signals get represented in the tree structure. Together they produce forests where effect estimation is not dominated by whichever predictor type happens to have more candidate split points.

---

## Computation time expectations

Rough timing for n = 400, p = 5, 3000 trees (single-threaded):

| Function | Time | What it does |
|----------|------|--------------|
| `infForest()` | ~5 sec | Fits 10 ranger forests (5 splits × 2 folds) |
| `effect()` | <1 sec | Point estimate, no forests fitted |
| `pasr_effect(variance_method="sandwich")` | ~2 sec | Point estimate + sandwich SE |
| `pasr_effect(variance_method="pasr")` | 1–3 min | Fits R×4 paired forests (~160 ranger fits at R=40) |
| `pasr_curve()` | 1–3 min | Same as above but extracts full curve per replicate |
| `pasr_predict()` | 1–5 min | PASR for all n predictions |

Times scale roughly linearly with n and num.trees. PASR time is dominated by ranger forest fitting, which scales as $O(B \cdot n \log n \cdot \text{mtry})$.

---

## Full API reference

### Fitting

```r
fit <- infForest(
  y ~ .,
  data = dat,
  num.trees = 3000,       # trees per forest (3000+ recommended)
  mtry = 5,               # candidates per split
  min.node.size = 10,     # terminal node size
  honesty.splits = 5,     # independent A/B partitions to average
  penalize = TRUE,        # standardized splitting criterion
  softmax = TRUE          # proportional variable selection
)
```

### Effects

```r
effect(fit, "x")                                      # binary or continuous
effect(fit, "x", at = c(0.10, 0.50, 0.90))            # multiple quantile contrasts
effect(fit, "x", at = c(30, 50, 70), type = "value")  # raw value contrasts
effect(fit, "x", bw = 50)                              # coarser grid (fewer intervals)
effect(fit, "x", subset = which(dat$z > 0))            # conditional on a subgroup
```

### Effect curves

```r
effect_curve(fit, "x")                        # point estimate curve (instant)
pasr_curve(fit, "x", verbose = TRUE)          # curve + bands + omnibus test
```

### Variance estimation

```r
# Sandwich SE (instant):
pasr_effect(fit, "trt", variance_method = "sandwich")

# PASR SE:
pasr_effect(fit, "trt", variance_method = "pasr", verbose = TRUE)

# Both (default):
pasr_effect(fit, "trt")
```

### Interactions

```r
int(fit, "x1", by = "x2")                             # x1 effect within x2 groups
int(fit, "x1", by = "x2", subset = which(dat$x3 == 1)) # three-way
int(fit, "x1", by = "x2",                             # continuous conditioning
    by_at = list(c(0.10, 0.25), c(0.75, 0.90)))
```

### Batch summary

```r
summary(fit, ~ x1 + x2 + x3)                          # main effects
summary(fit, ~ x1 * x2)                                # main + interaction
summary(fit, ~ x1[0.10, 0.50, 0.90] + x2)             # custom comparison points
```

### Prediction intervals

```r
pasr_predict(fit, verbose = TRUE)                      # at training points
pasr_predict(fit, newdata = test_data)                 # at new points
```

### Diagnostics

```r
ct_diagnose(fit)                                       # covariance floor diagnostic
```

---

## Citation

```bibtex
@article{oconnell2026rf,
  title={Random Forests as Statistical Procedures: Design, Variance, and Dependence},
  author={O'Connell, Nathaniel S.},
  journal={arXiv preprint arXiv:2602.13104},
  year={2026}
}

@article{oconnell2026infforest,
  title={Inference Forests: A Framework for Nonparametric Inference},
  author={O'Connell, Nathaniel S.},
  year={2026}
}
```
