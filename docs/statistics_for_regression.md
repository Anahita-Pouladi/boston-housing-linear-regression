# Statistics for Regression — Boston Housing Project

This guide covers the statistical concepts that are directly relevant to the Boston Housing Linear Regression project.

The goal is not to study statistics in isolation. Each topic is connected to how regression models are explored, diagnosed, interpreted, and evaluated in practice.

---

## 1. Why Statistics Matters in Regression

Machine learning focuses on prediction, while statistics helps us understand:

- How variables behave
- How variables relate to one another
- How uncertain our estimates may be
- Whether model assumptions are reasonable
- Whether model coefficients are stable and interpretable
- Whether observed patterns may be driven by noise

For Linear Regression, statistical reasoning is especially important because the model is both a predictive method and a classical statistical model.

---

## 2. Mean

The mean is the arithmetic average of a variable.

```text
Mean = Sum of values / Number of values
```

For a variable such as `MEDV`, the mean gives a single-number summary of the center of the distribution.

However, the mean is sensitive to extreme values.

### Why it matters in this project

The Boston Housing dataset contains observations near the upper end of the target distribution, including repeated values at `MEDV = 50`.

Because extreme or concentrated values can affect the mean, it is useful to compare the mean with the median.

---

## 3. Median

The median is the middle value after observations are sorted.

It is more resistant to extreme observations than the mean.

### Why it matters

When a distribution is skewed or contains outliers, the median can provide a more robust summary of central tendency.

For housing-related variables, comparing mean and median helps identify whether the distribution is symmetric or skewed.

---

## 4. Mean vs. Median

If:

```text
Mean ≈ Median
```

the distribution may be relatively symmetric.

If:

```text
Mean > Median
```

the distribution may be right-skewed.

If:

```text
Mean < Median
```

the distribution may be left-skewed.

This is only a diagnostic clue, not a complete test of distribution shape.

---

## 5. Variance

Variance measures how far observations spread around the mean.

Conceptually:

```text
Variance = Average squared distance from the mean
```

A larger variance means observations are more dispersed.

### Why variance matters in regression

Regression tries to explain variation in the target.

R², for example, is based on comparing unexplained variation with the total variation in the target.

---

## 6. Standard Deviation

Standard deviation is the square root of variance.

```text
Standard Deviation = √Variance
```

Unlike variance, standard deviation is expressed in the same unit as the original variable.

### Why it matters

Standard deviation is easier to interpret when comparing feature variability or understanding how dispersed a variable is around its mean.

---

## 7. Distribution

A distribution describes how values are spread across a variable.

Important characteristics include:

- Center
- Spread
- Skewness
- Tails
- Outliers
- Clusters
- Censoring or concentration at boundaries

### Boston Housing example

The target `MEDV` contains repeated observations at `50`.

This creates an upper-end concentration that should be considered when interpreting the distribution and model residuals.

The presence of this pattern does not automatically mean the target should be transformed.

---

## 8. Does the Target Need to Be Normally Distributed?

No.

A common misconception is:

> Linear Regression requires the target variable itself to be normally distributed.

That is not a general requirement for fitting a linear regression model.

Normality assumptions are more closely connected to the **residuals** when performing classical statistical inference such as:

- Confidence intervals
- Hypothesis tests
- t-tests for coefficients
- F-tests

For predictive modeling, the more important questions are whether the model generalizes and whether residual behavior suggests serious model misspecification.

---

## 9. Covariance

Covariance measures whether two variables tend to move together.

A positive covariance means they tend to move in the same direction.

A negative covariance means they tend to move in opposite directions.

A covariance near zero suggests little linear co-movement.

### Limitation

Covariance depends on the units of the variables, so its magnitude is difficult to compare across feature pairs.

That is one reason correlation is often easier to interpret.

---

## 10. Pearson Correlation

Pearson correlation measures the strength and direction of a linear relationship between two variables.

Its values range from:

```text
-1 to +1
```

Interpretation:

```text
+1  → perfect positive linear relationship
 0  → no linear relationship
-1  → perfect negative linear relationship
```

### Boston Housing example

In this project:

- `RM` shows a strong positive relationship with `MEDV`
- `LSTAT` shows a strong negative relationship with `MEDV`

Pearson correlation is used because the baseline model focuses on linear relationships.

### Important caution

Correlation does not imply causation.

A strong correlation can exist because of:

- Confounding variables
- Shared underlying factors
- Structural relationships
- Historical or sampling effects

---

## 11. Correlation vs. Covariance

Both describe how variables move together, but they differ in interpretation.

### Covariance

- Depends on measurement units
- Magnitude is not standardized

### Correlation

- Standardized
- Always between `-1` and `+1`
- Easier to compare across variable pairs

---

## 12. Outliers

An outlier is an observation that lies unusually far from the rest of the data.

Outliers can arise from:

- Data-entry errors
- Measurement errors
- Rare but valid observations
- Naturally heavy-tailed distributions

### Why outliers matter in Linear Regression

OLS minimizes squared errors.

Because residuals are squared, large errors receive disproportionately large penalties.

That means extreme observations can strongly influence:

- Regression coefficients
- Intercept
- Residual variance
- Evaluation metrics

---

## 13. IQR Method

The Interquartile Range is:

```text
IQR = Q3 - Q1
```

A common rule flags observations outside:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

### Important principle

The IQR method identifies potential outliers.

It does **not** automatically justify deleting, clipping, or replacing them.

In this project, outliers are flagged for investigation rather than automatically modified.

---

## 14. Residuals

A residual is:

```text
Residual = Actual - Predicted
```

Residuals are central to regression diagnostics.

A good residual pattern often looks like random scatter around zero.

Systematic patterns may suggest:

- Nonlinearity
- Heteroscedasticity
- Missing features
- Outliers
- Model misspecification

---

## 15. Residual Normality

Residual normality is often assessed using:

- Histogram
- KDE
- Q-Q plot
- Formal tests

### Q-Q Plot

A Q-Q plot compares residual quantiles with theoretical normal quantiles.

If points roughly follow the reference line, the residual distribution is approximately normal.

Deviations in the tails can indicate:

- Skewness
- Heavy tails
- Outliers

### Important caution

Perfect normality is rarely observed in real-world data.

The practical importance depends on the goal:

- Prediction
- Statistical inference
- Confidence intervals
- Hypothesis testing

---

## 16. Homoscedasticity

Homoscedasticity means the residual variance is approximately constant across the range of fitted values.

A desirable residual plot often looks like:

```text
random cloud around zero
```

A funnel shape may indicate heteroscedasticity.

### Why it matters

Heteroscedasticity can affect:

- Standard errors
- Confidence intervals
- Hypothesis tests

It does not necessarily make predictions unusable, but it changes how statistical inference should be interpreted.

---

## 17. Multicollinearity

Multicollinearity occurs when predictor variables are strongly related to each other.

Examples may include variables that capture similar structural or socioeconomic patterns.

### Why it matters

Multicollinearity can cause:

- Unstable coefficients
- Large standard errors
- Coefficient sign changes
- Difficulty interpreting individual variables

A model can still predict reasonably well even when coefficient interpretation becomes unstable.

---

## 18. Variance Inflation Factor (VIF)

VIF is a common diagnostic for multicollinearity.

Conceptually:

```text
VIF = 1 / (1 - R²)
```

where the R² comes from regressing one predictor on the remaining predictors.

Interpretation is context-dependent, but larger VIF values indicate greater redundancy.

### Important principle

VIF is a diagnostic.

It should not be used as an automatic rule such as:

```text
high VIF → always delete feature
```

Feature removal should consider:

- Domain meaning
- Predictive performance
- Interpretability goals
- Validation results
- Model stability

In the Boston Housing project, `RAD` and `TAX` are among the variables worth examining for multicollinearity.

---

## 19. Confidence Intervals

A confidence interval provides a range of plausible values for an estimated parameter under a statistical model.

For a coefficient, a confidence interval helps communicate uncertainty around the estimated effect.

A typical 95% confidence interval is conceptually:

```text
Estimate ± Margin of Error
```

### Important interpretation

A 95% confidence interval does not mean there is a 95% probability that the fixed true parameter lies inside one specific computed interval.

Instead, under repeated sampling and model assumptions, the procedure is designed to capture the true parameter about 95% of the time.

---

## 20. Hypothesis Testing

Hypothesis testing evaluates whether observed evidence is consistent with a specified null hypothesis.

For a regression coefficient, a common null hypothesis is:

```text
H₀: β = 0
```

which means the predictor has no linear association with the target after accounting for the other predictors in the model.

The alternative hypothesis is commonly:

```text
H₁: β ≠ 0
```

---

## 21. p-Value

A p-value measures how compatible the observed data is with the null hypothesis, assuming the null hypothesis and model assumptions hold.

A small p-value suggests the observed result would be unusual under the null hypothesis.

### What a p-value does NOT mean

A p-value is not:

- The probability that the null hypothesis is true
- The probability that the result happened by chance
- A measure of practical importance
- A direct measure of predictive usefulness

### Statistical vs. practical significance

A feature can be statistically significant but have a small practical effect.

A feature can also be predictively useful even if classical coefficient inference is not the main modeling objective.

---

## 22. Type I and Type II Errors

### Type I Error

Rejecting a true null hypothesis.

Also called a false positive.

### Type II Error

Failing to reject a false null hypothesis.

Also called a false negative.

These concepts are important when interpreting hypothesis tests.

---

## 23. Statistical Significance vs. Predictive Performance

Statistical inference and machine learning prediction are related but not identical goals.

A statistical analysis may ask:

```text
Is this coefficient distinguishable from zero?
```

A machine learning analysis may ask:

```text
How accurately does the model predict unseen observations?
```

A strong project should understand both perspectives.

This Boston Housing project primarily emphasizes predictive modeling while also using statistical diagnostics for interpretation.

---

## 24. R² from a Statistical Perspective

R² measures the proportion of observed target variance explained by the fitted model relative to a mean baseline.

Conceptually:

```text
R² = 1 - (Unexplained Variation / Total Variation)
```

An R² near 1 indicates that much of the sample variation is explained by the model.

An R² near 0 means the model does not improve much over predicting the target mean.

R² can also be negative on test data if predictions are worse than a mean baseline.

---

## 25. Adjusted R²

Regular R² cannot decrease simply because more predictors are added to a fitted model.

Adjusted R² introduces a penalty related to the number of predictors.

That makes it more useful when comparing linear models with different feature counts.

In this project:

```text
R² ≈ 0.659
Adjusted R² ≈ 0.609
```

The difference reflects the adjustment for model complexity.

---

## 26. Sampling Variability

Model results depend on the particular sample used for training and testing.

A single train/test split may therefore produce a result that is unusually high or low.

Cross-validation helps quantify performance across multiple splits.

In this project, 5-fold cross-validation produced approximately:

| Metric | Mean Result |
|---|---:|
| R² | ~0.708 |
| MAE | ~3.415 |
| RMSE | ~4.912 |

This gives a broader picture of generalization than a single split alone.

---

## 27. Bias and Variance

### Bias

Bias refers to systematic error from simplifying assumptions.

A very simple model may have high bias.

### Variance

Variance refers to how sensitive a model is to changes in the training data.

A very flexible model may have high variance.

### Bias-Variance Tradeoff

A useful model balances:

```text
enough flexibility to learn signal
+
enough simplicity to generalize
```

Linear Regression is often a low-variance, interpretable baseline, although it can underfit nonlinear relationships.

---

## 28. Why Statistical Diagnostics Should Be Contextual

There is rarely a single universal rule such as:

```text
p < 0.05 → feature is important
VIF > threshold → delete feature
non-normal target → transform target
outlier detected → remove row
```

A better approach is:

```text
Diagnose
   ↓
Understand the cause
   ↓
Consider modeling objective
   ↓
Validate the decision
```

This project follows that principle.

---

## 29. Project-Specific Statistical Takeaways

For the Boston Housing Linear Regression project:

- The dataset contains 506 observations.
- The target `MEDV` has no missing values.
- Six predictor columns contain missing observations.
- Missing predictor values are handled with median imputation inside a Pipeline.
- Pearson correlation is used to study linear relationships.
- `RM` is positively associated with `MEDV`.
- `LSTAT` is negatively associated with `MEDV`.
- IQR is used for outlier flagging, not automatic removal.
- Residual plots and a Q-Q plot are used for model diagnostics.
- VIF is used to investigate multicollinearity.
- Cross-validation is used to reduce dependence on a single data split.
- Predictive metrics and statistical diagnostics are interpreted together.

---

## 30. Interview Questions

### Q1. What is the difference between mean and median?

The mean is the arithmetic average and is more sensitive to extreme values, while the median is the middle ordered value and is more robust to outliers.

### Q2. What is the difference between variance and standard deviation?

Variance measures average squared deviation from the mean, while standard deviation is the square root of variance and is expressed in the original unit.

### Q3. What does covariance measure?

It measures whether two variables tend to move together, but its magnitude depends on measurement units.

### Q4. Why is correlation easier to interpret than covariance?

Correlation is standardized between -1 and +1, making it easier to compare relationships.

### Q5. Does correlation imply causation?

No.

### Q6. Why do outliers affect Linear Regression strongly?

Because OLS squares residuals, so large errors receive disproportionately large weight.

### Q7. What is multicollinearity?

It is strong dependence among predictor variables, which can make coefficient estimates unstable.

### Q8. What does VIF measure?

It estimates how much the variance of a regression coefficient is inflated because of multicollinearity.

### Q9. Does a high VIF always mean a feature should be removed?

No. VIF is a diagnostic and should be interpreted in context.

### Q10. Does Linear Regression require a normally distributed target?

No. Normality assumptions are mainly related to residual-based inference.

### Q11. What is a p-value?

It measures how compatible the observed data is with the null hypothesis under the assumed statistical model.

### Q12. What is the difference between statistical significance and practical significance?

Statistical significance concerns evidence against a null hypothesis, while practical significance concerns whether the effect is large or useful enough to matter.

### Q13. Why use confidence intervals?

They communicate uncertainty around an estimated parameter rather than reporting only a single point estimate.

### Q14. What is homoscedasticity?

It means residual variance is approximately constant across fitted values.

### Q15. Why use cross-validation?

It helps estimate generalization performance across multiple data splits and reduces reliance on one particular train/test split.

---

## 31. Key Takeaways

Statistics strengthens regression modeling by helping answer questions that performance metrics alone cannot answer.

A practical statistical workflow is:

```text
Describe the data
       ↓
Understand distributions
       ↓
Explore correlations
       ↓
Flag unusual observations
       ↓
Fit the regression model
       ↓
Analyze residuals
       ↓
Check multicollinearity
       ↓
Quantify predictive performance
       ↓
Assess uncertainty where relevant
       ↓
Interpret results in context
```

The most important principle is to use statistical tools as diagnostics and reasoning aids, not as automatic rules.
