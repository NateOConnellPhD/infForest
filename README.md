# infForest

> Nonparametric inference from random forests — effects, predictions, and uncertainty quantification

**infForest** is an R package that turns random forests into full inferential procedures. It provides effect estimates for any predictor (continuous, binary, or categorical), nonlinear effect curves, interaction detection, prediction intervals, confidence intervals for predicted probabilities, and variance decomposition diagnostics — all without functional form assumptions.

## What is this framework?

This is a complement to regression, not a replacement. Regression answers: *what do these data say about parameter estimates given my imposed assumptions about the functional form?* Inference forests answer: *what do the data themselves say about the effects without any imposed restrictions?*

When both agree, the parametric assumptions are consistent with the data. When they disagree, the finite sample doesn't match the imposed structure. Both answers are useful.

**Nonparametric.** No linearity, no additivity, no link functions. The forest discovers the functional form from the data. For binary outcomes, predicted probabilities live on [0, 1] naturally — no logit link, no linearity on the log-odds scale.

**Asymptotically unbiased.** As sample size grows, the estimates converge to the true effects. In finite samples, the estimates are provably conservative — they underestimate effects, never overestimate. This conservative bias decreases with sample size and with model simplicity (fewer predictors, less complexity). If you see an effect, it's there. If you don't, it may be attenuated by the nonparametric smoothing.

**Confounding adjustment.** Effects are adjusted for all other variables in the model through the AIPW framework. The propensity model (estimated by ridge regression) corrects for confounding, analogous to including covariates in a regression model. The correction is doubly robust: the estimate is consistent if either the forest predictions or the propensity model is correctly specified.

**Procedural estimand.** The estimates target the expected value of the estimate if we could repeatedly draw new outcome values (Y) from the same covariate population and apply the same procedure each time. The confidence interval covers this procedural target — the natural analogue of what regression targets.

**No p-values by design.** The package focuses on effect sizes and precision. Every estimate comes with a standard error and confidence interval at a user-specified level (default 95%). The estimate and SE are provided if p-values are needed downstream. The deliberate omission reflects a focus on practical significance over null hypothesis testing.

### Conservative bias and model complexity

The finite-sample conservative bias (attenuation toward zero) arises from the forest's nonparametric smoothing. Two factors control its magnitude:

- **Sample size.** More data → smaller leaves → less smoothing → less attenuation.
- **Model complexity.** More predictors spread the forest's splitting budget, increasing effective smoothing per variable. Reducing irrelevant predictors concentrates the forest on important dimensions and reduces attenuation.

Variable selection is an active area of development for this framework. Guidance on principled variable selection for inference forests is forthcoming. For now, reducing the predictor set to relevant variables will improve finite-sample performance.


## Installation

```r
# Requires the inf.ranger fork of ranger
devtools::install_github("NateOConnellPhD/inf.ranger")
devtools::install_github("NateOConnellPhD/infForest")
```

### About inf.ranger

infForest uses [ranger](https://github.com/imbs-hl/ranger) as its forest engine. The `inf.ranger` package is a minimal fork of ranger with two targeted modifications to the split selection code — the tree-growing algorithm, prediction machinery, and all other ranger internals are unchanged.

**`penalize.split.competition`** — At each node, CART selects the variable with the largest impurity reduction. But continuous variables evaluate many more candidate split points than binary variables, giving them a structural advantage. The standardized criterion subtracts the expected search advantage from each variable's best split score, producing a level comparison across variable types. The correction is closed-form and adds negligible computation.

**`softmax.split`** — Standard CART selects the single best variable at each node (argmax). Softmax replaces this with probabilistic selection: variables are chosen with probability proportional to their penalized criterion scores. This increases the inclusion rate for variables with moderate but real signal. Requires `penalize.split.competition = TRUE`.

Both modifications operate only at the moment of variable selection within each node. Everything downstream — split point selection, daughter node assignment, leaf predictions, prediction — is identical to standard ranger.


## Part 1: Pointwise Predictions and Prediction Intervals

Pointwise predictions with variance estimates and prediction intervals are a standalone contribution applicable to **any fitted ranger model**. You do not need `infForest()` for this — in fact, using `infForest()` for prediction is not recommended because it halves the data for honest estimation. For pure prediction, fit ranger on the full dataset.

The PASR (Procedure-Aligned Synthetic Resampling) framework is the only approach available for variance estimation of pointwise random forest predictions. It currently supports `ranger` models. Support for other forest packages (`randomForest`, `aorsf`, etc.) is planned for future releases.

### Prediction intervals (continuous)

```r
library(inf.ranger)
library(infForest)

set.seed(42)
n <- 1000
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
y <- 2 * x1 + sin(x2) + rnorm(n)
dat <- data.frame(x1, x2, x3, y)

# Fit ranger directly on full data — not infForest
rf <- ranger(y ~ ., data = dat, num.trees = 5000)

# PASR prediction intervals
pi <- pasr_predict_ranger(rf, dat, R_max = 50, verbose = TRUE)
head(pi[, c("f_hat", "se", "ci_lower", "ci_upper", "pi_lower", "pi_upper")])

# Coverage check
cat("PI coverage:", mean(y >= pi$pi_lower & y <= pi$pi_upper), "\n")
```

### Probability estimates with confidence intervals (binary)

For binary outcomes, predicted probabilities live on [0, 1] naturally. No logit link, no back-transformation.

```r
y_bin <- factor(rbinom(n, 1, plogis(x1 + 0.5 * x2)))
dat_bin <- data.frame(x1, x2, x3, y = y_bin)

rf_bin <- ranger(y ~ ., data = dat_bin, num.trees = 5000, probability = TRUE)
pi_bin <- pasr_predict_ranger(rf_bin, dat_bin, R_max = 50)
head(pi_bin[, c("f_hat", "se", "ci_lower", "ci_upper")])
```

### Predictions at new data

```r
newdata <- data.frame(x1 = rnorm(20), x2 = rnorm(20), x3 = rnorm(20))
pasr_predict_ranger(rf, dat, newdata = newdata, R_max = 50)
```

For details on the PASR framework, see [O'Connell (2026)](https://arxiv.org/abs/2602.13104).


## Part 2: Nonparametric Effect Estimation and Inference

This is the core of `infForest`: estimating population-averaged effects for any predictor — continuous, binary, or categorical — with valid confidence intervals and no parametric assumptions. All language here describes adjusted effect estimates, analogous to regression coefficients. No causal claims are made.

### Fitting the model

```r
set.seed(42)
n <- 1000
x1 <- rnorm(n)
x2 <- rnorm(n)
trt <- rbinom(n, 1, 0.4)
x4 <- 0.5 * x2 + rnorm(n) * sqrt(0.75)
noise <- rnorm(n)
group <- factor(sample(c("A", "B", "C"), n, replace = TRUE, prob = c(0.5, 0.3, 0.2)))

mu <- 0.8 * sin(1.5 * x1) + 0.4 * x2 + 0.3 * trt + 0.2 * x2 * trt +
      ifelse(group == "A", -0.10, ifelse(group == "B", 0.25, 0.40))

y <- mu + (0.5 + 0.2 * abs(x1)) * rnorm(n)
dat <- data.frame(x1, x2, trt, x4, noise, group, y)

fit <- infForest(y ~ ., data = dat, num.trees = 5000,
                 penalize = TRUE, softmax = TRUE)
```

The data frame must be complete — either complete cases or imputed.

### Binary effects

```r
effect(fit, "trt")
#> Inference Forest Effect Estimate
#>   Variable:    trt
#>   Type:        binary
#>   Estimate:    0.2892
#>   SE (sandwich): 0.0412  |  95% CI: [0.2084, 0.3700]
```

### Continuous effects

For continuous variables, `infForest` estimates effects through a nonparametric effect curve — a grid of local slopes that captures the shape of the relationship without assuming linearity.

```r
# Default: per-unit slope from Q25 to Q75
effect(fit, "x2")

# Multiple quantile contrasts — returns all pairwise
effect(fit, "x2", at = c(0.10, 0.50, 0.90))
#>   Contrasts (per unit):
#>     Q50 - Q10  [-1.383, -0.068]:  0.8810
#>     Q90 - Q10  [-1.383, 1.182]:   0.8332
#>     Q90 - Q50  [-0.068, 1.182]:   0.7830

# Raw value contrasts
effect(fit, "x2", at = c(-1, 0, 1), type = "value")

# Wider bandwidth — fewer grid intervals, faster, coarser curve
effect(fit, "x2", bw = 50)
```

**How the grid works.** The effect curve is estimated over a fine grid of evaluation points between the comparison values. At each grid interval, a local slope is computed via the AIPW estimator. The reported per-unit effect is the integrated average of these local slopes. This avoids assuming linearity between the comparison points — a nonlinear relationship yields a different integrated slope than a linear one.

The grid resolution is controlled by `bw` (bandwidth — observations per grid interval). The default `bw = 20` balances computational cost with curve fidelity.

- **Finer grid** (smaller `bw`, e.g. 10): more grid points, slower computation, captures finer nonlinearity. Best for detailed curve inspection.
- **Coarser grid** (larger `bw`, e.g. 50): fewer grid points, faster computation. Approaches a linear approximation between comparison points.
- **Very coarse** (e.g. `bw = n/2`): effectively a single interval — equivalent to assuming linearity between the two points. This is the fastest option when linearity is a reasonable assumption.

### Effect curves

The effect curve shows the full nonparametric relationship, relative to a reference point. This is a major feature: the forest discovers the functional shape without being told it exists.

```r
ec <- effect_curve(fit, "x1", bw = 20, q_lo = 0.02, q_hi = 0.98)
plot(ec)

# Compare to DGM truth (0.8 * sin(1.5 * x))
x1_ref <- ec$ref
curve(0.8 * sin(1.5 * x) - 0.8 * sin(1.5 * x1_ref),
      add = TRUE, col = "red", lty = 2, lwd = 2)
```

Regression would need `sin(x1)` specified in the formula. The inference forest discovers the shape from the data.

### Categorical effects

Categorical variables are handled through pairwise contrasts within the subpopulation of observations belonging to the two compared levels.

```r
# All pairwise contrasts
effect(fit, "group")

# Specific contrast
effect(fit, "group", at = c("A", "B"))
```

### Interactions

```r
# Effect of x2 within levels of trt
int(fit, "x2", by = "trt")

# Continuous conditioning variable with custom bands
int(fit, "x2", by = "x1", by_at = list(c(0.10, 0.25), c(0.75, 0.90)))
```

### Summary — multiple effects at once

The `summary()` function estimates multiple effects in a single call. Binary variables are batched through an optimized multi-variable scorer.

The bracket notation controls comparison points directly in the formula. For continuous variables, brackets specify quantiles (or raw values with `type = "value"`). For categorical variables, brackets specify level names.

```r
# Main effects with defaults
summary(fit, ~ trt + x2 + noise + group)

# Custom quantile contrasts for x2, specific categorical contrast
summary(fit, ~ trt + x2[.10, .90] + group["A", "B"])

# Main effects plus interaction
summary(fit, ~ trt + x2*trt)

# Interaction only
summary(fit, ~ x2:trt)
```

### Binary outcomes

For binary outcomes, the forest estimates probabilities directly on [0, 1]. Effect estimates are on the probability scale.

```r
y_bin <- factor(rbinom(n, 1, plogis(mu)), levels = c(0, 1))
dat_bin <- data.frame(x1, x2, trt, x4, noise, group, y = y_bin)

fit_bin <- infForest(y ~ ., data = dat_bin, num.trees = 5000,
                     penalize = TRUE, softmax = TRUE)

# Treatment effect on probability scale
effect(fit_bin, "trt")

# Predicted probabilities with CIs
pi_bin <- pasr_predict(fit_bin, R_max = 50)
head(pi_bin[, c("f_hat", "se", "ci_lower", "ci_upper")])
```


## Variance estimation

Two variance estimators are available:

- **Sandwich** (default): derived from the AIPW influence function scores. Fast, computed alongside the point estimate at no additional cost. Applicable to marginal effect estimates and contrasts.
- **PASR**: based on Procedure-Aligned Synthetic Resampling. More thorough, slower. The only option for pointwise prediction variance. Provides a diagnostic ratio `rho_V` when both are computed.

```r
effect(fit, "trt", variance = "sandwich")   # default
effect(fit, "trt", variance = "pasr")       # slower, more thorough
effect(fit, "trt", variance = "both")       # comparison with rho_V diagnostic
```

### Confidence level

```r
effect(fit, "trt", alpha = 0.10)  # 90% CI
effect(fit, "trt", alpha = 0.01)  # 99% CI
```


## Full API reference

### Fitting

`infForest()` fits the honest cross-fitted forest. Data is internally split into build and estimation folds; trees are grown on the build fold, and all subsequent estimates use only estimation-fold outcomes. The forest cache (precomputed tree structure, leaf means, and observation routing) is built at fit time to accelerate all downstream calls.

```r
fit <- infForest(
  y ~ .,
  data = dat,
  num.trees = 5000,        # trees per forest (5000 recommended)
  mtry = 5,                # candidates per split
  min.node.size = 10,      # terminal node size
  honesty.splits = 5,      # independent A/B partitions to average
  penalize = TRUE,         # standardized splitting criterion
  softmax = TRUE           # proportional variable selection
)
```

### Effects

`effect()` estimates the population-averaged effect of a single predictor, adjusted for all other variables in the model. This is the primary function for individual effect estimation.

For **binary** predictors, the estimate is the average difference in predicted outcomes between the two levels (analogous to a regression coefficient for a binary variable).

For **continuous** predictors, the estimate is the per-unit slope computed from a nonparametric effect curve between two (or more) comparison points. When `at` contains 3+ values, all pairwise contrasts are returned. The grid of local slopes avoids imposing linearity — the integrated slope between Q10 and Q90 reflects the actual shape of the relationship.

For **categorical** predictors, all pairwise contrasts are returned by default, or specific level comparisons can be requested with `at = c("A", "B")`. Each contrast is estimated within the subpopulation of observations belonging to those two levels.

```r
effect(fit, "x")                                      # binary or continuous (defaults)
effect(fit, "x", at = c(0.10, 0.50, 0.90))            # 3 quantiles → 3 pairwise contrasts
effect(fit, "x", at = c(30, 50, 70), type = "value")  # raw value contrasts
effect(fit, "x", bw = 50)                              # coarser grid (faster)
effect(fit, "x", alpha = 0.10)                         # 90% CI
effect(fit, "x", variance = "both")                    # sandwich + PASR comparison
effect(fit, "cat_var")                                 # all pairwise categorical contrasts
effect(fit, "cat_var", at = c("A", "C"))               # specific categorical contrast
effect(fit, "x", subset = which(dat$z > 0))            # conditional on subgroup
```

### Effect curves

`effect_curve()` traces the full nonparametric relationship between a continuous predictor and the outcome, relative to a reference point (default: the median). This is the most informative view of a continuous effect — it shows the shape without summarizing to a single slope. The curve captures nonlinear relationships (sine curves, thresholds, plateaus) that a single per-unit slope would miss.

```r
ec <- effect_curve(fit, "x")
ec <- effect_curve(fit, "x", bw = 30, q_lo = 0.05, q_hi = 0.95)
plot(ec)
```

### Interactions

`int()` estimates how the effect of one variable differs across levels or regions of another variable. For a binary conditioning variable, the effect is estimated within each level and the difference is reported. For a continuous conditioning variable, the effect is estimated within quantile bands and the difference between the high and low bands is reported.

This is analogous to an interaction term in regression, but without assuming the interaction is multiplicative or linear.

```r
int(fit, "x1", by = "x2")                             # x1 effect within x2 groups
int(fit, "x1", by = "x2", subset = which(dat$x3 == 1)) # three-way
int(fit, "x1", by = "x2",                             # continuous conditioning
    by_at = list(c(0.10, 0.25), c(0.75, 0.90)))
```

### Summary

`summary()` estimates multiple effects in a single call. For binary variables, an optimized multi-variable scorer shares tree-walking across variables, making `summary()` substantially faster than repeated `effect()` calls. Use `summary()` when estimating several effects — it is the recommended entry point for routine analysis.

The **bracket notation** controls comparison points directly in the formula:
- **Continuous variables:** `x2[.10, .90]` computes the per-unit slope from Q10 to Q90 (quantiles by default, raw values with `type = "value"`)
- **Categorical variables:** `group["A", "B"]` computes the A vs B contrast
- **No brackets:** defaults are used (Q25 to Q75 for continuous, all pairwise for categorical, 1 vs 0 for binary)
- **Interactions:** `x1*x2` expands to main effects plus interaction; `x1:x2` is interaction only

```r
summary(fit, ~ x1 + x2 + x3)                          # main effects
summary(fit, ~ x1 * x2)                                # main + interaction
summary(fit, ~ x1[0.10, 0.50, 0.90] + x2)             # custom comparison points
summary(fit, ~ x1:x2[0.10,0.25,0.75,0.90])            # custom conditioning bands
summary(fit, ~ trt + group["A", "B"])                   # categorical contrast
summary(fit, ~ trt + x2, alpha = 0.10)                  # 90% CIs for all terms
```

### PASR

`pasr_predict()` computes pointwise prediction intervals using Procedure-Aligned Synthetic Resampling. This is the only method available for variance estimation of individual forest predictions (the sandwich estimator applies only to marginal effect estimates). For continuous outcomes, both confidence intervals (for the conditional mean) and prediction intervals (for a new observation) are provided. For binary outcomes, confidence intervals for the predicted probability are provided.

`ct_diagnose()` estimates the covariance floor — the irreducible prediction uncertainty that persists even with infinite trees — and provides diagnostic plots.

```r
pasr_predict(fit, verbose = TRUE)                      # pointwise prediction intervals
pasr_predict(fit, newdata = test_data)                 # at new points
ct_diagnose(fit)                                       # covariance floor diagnostic
```


## Design parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num.trees` | 5000 | More trees → lower MC variance. 5000 recommended for inference. |
| `honesty.splits` | 5 | Independent fold assignments averaged. At large n (>2000), 2-3 is sufficient. |
| `penalize` | TRUE | Corrects variable selection bias. Always use for inference. |
| `softmax` | FALSE | Proportional variable selection. Set TRUE for weak signals. |
| `min.node.size` | 10 | Controls resolution. Smaller → finer conditioning, more variance per leaf. |
| `replace` | FALSE | Sampling without replacement maximizes effective sample size per fold. |
| `bw` | 20 | Grid bandwidth for continuous effects. Larger → faster, coarser. |
| `alpha` | 0.05 | Confidence level for intervals (1 - alpha). |


## How it works

### Honest estimation
Data is split into build and estimation folds. Trees are grown on the build fold; all effect estimates use only estimation-fold outcomes. This eliminates adaptive bias. Multiple independent splits are averaged to reduce fold-assignment variance.

### AIPW debiasing
Forest prediction contrasts are biased because trees that split on a variable absorb its signal into the tree structure. The AIPW correction adds propensity-weighted honest residuals, making the bias the *product* of the prediction error and the propensity error — small even when both models are moderately wrong.

### Standardized splitting and softmax selection
Implemented in `inf.ranger`. The standardized criterion corrects variable selection bias from the search advantage; softmax selection ensures moderate signals get represented in the tree structure.

### PASR
The covariance floor captures irreducible dependence between trees sharing training data. PASR estimates it by generating synthetic outcomes from a fitted nuisance model, refitting paired forests on each synthetic dataset with shared fold assignments, and computing the cross-covariance. For effect functionals, scalar-first PASR operates on the effect estimate directly.


## Scope and limitations

- **No time series or clustered data.** The framework assumes independent observations. Extensions are future work.
- **No p-values by design.** Effect estimates and confidence intervals are provided. The SE is available for downstream computation.
- **Variable selection.** Principled methods for inference forests are under active development. Reducing the predictor set improves finite-sample performance.


## References

O'Connell, N.S. (2026). Random Forests as Statistical Procedures: Design, Variance, and Dependence. *arXiv:2602.13104*. [[paper]](https://arxiv.org/abs/2602.13104)

O'Connell, N.S. (2026). Inference Forests: A General Framework for Nonparametric Inference. *In preparation*.

## Citation

```bibtex
@article{oconnell2026rf,
  title={Random Forests as Statistical Procedures: Design, Variance, and Dependence},
  author={O'Connell, Nathaniel S.},
  journal={arXiv preprint arXiv:2602.13104},
  year={2026}
}

@article{oconnell2026infforest,
  title={Inference Forests: A General Framework for Nonparametric Inference},
  author={O'Connell, Nathaniel S.},
  year={2026}
}
```
