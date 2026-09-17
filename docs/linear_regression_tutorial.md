# Linear Regression — Practical Tutorial

This tutorial explains the core ideas behind Linear Regression and connects them directly to the Boston Housing project in this repository.

The goal is not only to know how to call `LinearRegression()` in Scikit-learn, but to understand what the model is doing, how to evaluate it, which assumptions matter, and when a linear model is or is not a sensible choice.

---

## 1. What Is Regression?

Regression is a supervised machine learning task used when the target variable is numerical and continuous.

Examples include predicting:

- House prices
- Sales revenue
- Temperature
- Delivery time
- Customer lifetime value

In this project, the target is:

```text
MEDV
```

`MEDV` represents the median value of owner-occupied homes in thousands of dollars.

Because `MEDV` is numerical and continuous, this is a regression problem.

---

## 2. Feature vs. Target

In supervised learning, the dataset is usually separated into:

- **Features (`X`)** — the input variables used to make predictions
- **Target (`y`)** — the value the model tries to predict

For the Boston Housing project:

```text
X = CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
y = MEDV
```

For example, `RM` is the average number of rooms per dwelling, while `MEDV` is the target housing value.

---

## 3. What Is Linear Regression?

Linear Regression models the relationship between the target and one or more features using a linear equation.

For one feature:

```text
ŷ = β₀ + β₁x
```

For multiple features:

```text
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

Where:

- `ŷ` = predicted target value
- `β₀` = intercept
- `β₁ ... βₚ` = model coefficients
- `x₁ ... xₚ` = feature values

The Boston Housing project uses **multiple linear regression** because it contains multiple predictor variables.

---

## 4. Intercept

The intercept is the predicted value of the target when all input features are zero.

In the equation:

```text
ŷ = β₀ + β₁x₁ + ... + βₚxₚ
```

`β₀` is the intercept.

The intercept is mathematically necessary in most regression models, but its real-world interpretation may not always be meaningful if a zero value for every feature is unrealistic.

---

## 5. Coefficients

Each coefficient represents the expected change in the predicted target associated with a one-unit increase in that feature, while holding the other features constant.

For example, if the fitted coefficient for `RM` were positive, then — all else equal — increasing the average number of rooms would increase the model's predicted value of `MEDV`.

Important:

> A regression coefficient describes the fitted model relationship. It does not automatically establish causation.

Coefficient interpretation can also become unstable when predictors are strongly correlated with one another.

---

## 6. Residuals

A residual is the difference between an observed value and the model prediction:

```text
Residual = Actual Value - Predicted Value
```

If:

```text
Actual MEDV = 25
Predicted MEDV = 22
```

then:

```text
Residual = 3
```

Residual analysis is important because Linear Regression should not be evaluated only by a single accuracy metric.

Patterns in residuals can reveal:

- Nonlinearity
- Unequal error variance
- Outliers
- Missing structure
- Potential model misspecification

---

## 7. Ordinary Least Squares (OLS)

Linear Regression commonly estimates its coefficients using **Ordinary Least Squares**.

OLS chooses the coefficients that minimize the sum of squared residuals:

```text
Σ(yᵢ - ŷᵢ)²
```

Squaring the residuals has two important effects:

1. Positive and negative errors cannot cancel each other out.
2. Large errors receive greater penalties than small errors.

Scikit-learn's standard `LinearRegression` model uses an ordinary least-squares solution.

---

## 8. Why Split the Data?

If a model is evaluated on the same observations used for training, its performance can appear better than its true performance on unseen data.

A common workflow is therefore:

```text
Full Dataset
     ↓
Train Set + Test Set
     ↓
Fit on Train Set
     ↓
Evaluate on Test Set
```

In this project, the train/test split is reproducible because a fixed `random_state` is used.

The test set should remain unseen during model fitting.

---

## 9. Why Use a Pipeline?

The Boston Housing dataset contains missing predictor values.

Instead of manually filling missing values before the train/test split, preprocessing is included inside a Scikit-learn `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", LinearRegression())
])
```

This is important because preprocessing steps should learn their parameters from the training data rather than from the full dataset.

That helps reduce **data leakage**.

---

## 10. Does Linear Regression Require Feature Scaling?

For ordinary unregularized Linear Regression, feature scaling is not required to obtain valid predictions.

This project intentionally keeps the original feature units because that makes coefficients easier to interpret.

Scaling becomes much more important for models and techniques such as:

- Ridge Regression
- Lasso Regression
- Elastic Net
- K-Nearest Neighbors
- Support Vector Machines
- Principal Component Analysis

A common mistake is to scale a new observation even though the model itself was trained on unscaled data. Training and inference must always use the same preprocessing workflow.

Using a Pipeline helps enforce that consistency.

---

## 11. Linear Regression Assumptions

Linear Regression is often discussed with several assumptions. These assumptions mainly concern the relationship between the predictors and the errors, not whether the target variable itself looks perfectly normal.

### 11.1 Linearity

The expected relationship between the predictors and target should be reasonably linear.

Scatter plots and residual plots can help assess this.

### 11.2 Independent Errors

Residuals should not show strong dependence across observations.

This issue is especially important in time-series, spatial, or repeated-measures data.

### 11.3 Homoscedasticity

Residual variance should remain reasonably constant across the fitted-value range.

A funnel-shaped residual plot may indicate heteroscedasticity.

### 11.4 Residual Normality

For classical statistical inference, residuals are commonly expected to be approximately normal.

Important:

> Linear Regression does **not** require the target variable itself to be normally distributed.

Transforming the target only because it is not normal is therefore not automatically necessary.

### 11.5 Limited Multicollinearity

Predictors should not be excessively redundant with one another.

High multicollinearity can make coefficient estimates unstable and harder to interpret.

In this project, Variance Inflation Factor (VIF) is used as a diagnostic.

---

## 12. Correlation vs. Regression

Correlation measures the strength and direction of association between variables.

Regression builds a predictive equation.

A high correlation between a feature and the target can be informative, but it does not guarantee that the feature will have the same importance inside a multivariable regression model.

This is because regression coefficients are estimated while accounting for the other predictors.

The project uses **Pearson correlation** because the baseline analysis focuses on linear relationships.

---

## 13. Multicollinearity

Multicollinearity occurs when predictors are strongly related to one another.

Possible consequences include:

- Unstable coefficients
- Large coefficient changes across samples
- Difficulty interpreting individual predictors
- Inflated coefficient uncertainty

A common diagnostic is the **Variance Inflation Factor (VIF)**.

High VIF does not automatically mean a feature must be deleted.

Feature removal should depend on:

- Modeling objective
- Domain meaning
- Predictive performance
- Interpretability requirements
- Stability across validation

In the Boston Housing project, variables such as `RAD` and `TAX` are examined for multicollinearity, but features are not removed automatically.

---

## 14. Outliers

Outliers can strongly influence an OLS regression because squared errors give large residuals extra weight.

The IQR rule is one way to flag potential outliers:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

However, detecting an outlier does not automatically justify removing or clipping it.

In this project, IQR-based analysis is used for diagnosis rather than automatic modification of the dataset.

---

## 15. Overfitting and Underfitting

### Underfitting

A model underfits when it is too simple to capture important structure in the data.

Possible signs:

- Poor training performance
- Poor test performance

### Overfitting

A model overfits when it learns the training data too closely and does not generalize well.

Possible signs:

- Very strong training performance
- Much weaker validation or test performance

Linear Regression is relatively simple, but overfitting can still happen when:

- There are many predictors relative to the number of observations
- Polynomial features are added aggressively
- The data contains noisy or redundant variables

Cross-validation helps assess generalization more reliably.

---

## 16. Cross-Validation

A single train/test split can produce results that depend on one particular random split.

K-fold cross-validation repeatedly trains and evaluates the model on different partitions of the data.

For 5-fold cross-validation:

```text
Dataset
├── Fold 1
├── Fold 2
├── Fold 3
├── Fold 4
└── Fold 5
```

Each fold acts as validation data once while the other folds are used for training.

In this project, 5-fold shuffled cross-validation provides a more robust estimate of model performance.

Approximate mean cross-validation results:

| Metric | Mean Result |
|---|---:|
| R² | ~0.708 |
| MAE | ~3.415 |
| RMSE | ~4.912 |

---

## 17. Regression Evaluation Metrics

### MAE — Mean Absolute Error

MAE is the average absolute prediction error.

Conceptually:

```text
MAE = average(|Actual - Predicted|)
```

Advantages:

- Easy to interpret
- Same unit as the target
- Less sensitive to large errors than MSE

Lower is better.

---

### MSE — Mean Squared Error

MSE averages squared prediction errors.

```text
MSE = average((Actual - Predicted)²)
```

Because errors are squared, large errors receive stronger penalties.

Lower is better.

---

### RMSE — Root Mean Squared Error

RMSE is the square root of MSE.

```text
RMSE = √MSE
```

Its main advantage is that it returns to the same unit as the target.

Lower is better.

---

### R² — Coefficient of Determination

R² measures how much of the target variance is explained by the model relative to a simple mean baseline.

Interpretation example:

```text
R² = 0.66
```

means the fitted model explains roughly 66% of the target variance in that evaluation sample.

Higher is generally better, but R² should not be used alone.

---

### Adjusted R²

Regular R² does not penalize the model simply for adding more predictors.

Adjusted R² accounts for the number of predictors and sample size, making it more useful when comparing linear models with different numbers of features.

---

## 18. Boston Housing Baseline Results

Approximate test-set performance from this project:

| Metric | Result |
|---|---:|
| MAE | ~3.15 |
| MSE | ~24.98 |
| RMSE | ~5.00 |
| R² | ~0.659 |
| Adjusted R² | ~0.609 |

These values establish a baseline rather than claiming that Linear Regression is the best possible model for the dataset.

---

## 19. Linear vs. Polynomial Regression

Linear Regression assumes the prediction is linear in the model coefficients.

If the relationship between a predictor and target is curved, polynomial features can allow the model to represent nonlinear patterns.

For example:

```text
Linear:
y = β₀ + β₁x

Polynomial:
y = β₀ + β₁x + β₂x²
```

Polynomial Regression can be more flexible, but increasing polynomial degree also increases the risk of overfitting and multicollinearity.

It should therefore be evaluated using validation or cross-validation rather than chosen only because it fits the training data better.

---

## 20. Regularization

Regularization adds a penalty to the regression objective.

Its main goals can include:

- Reducing coefficient magnitude
- Improving stability
- Managing multicollinearity
- Reducing overfitting

### Ridge Regression

Ridge uses an L2 penalty.

It tends to shrink coefficients toward zero without usually making them exactly zero.

### Lasso Regression

Lasso uses an L1 penalty.

It can shrink some coefficients exactly to zero, which may act as a form of feature selection.

### Elastic Net

Elastic Net combines L1 and L2 penalties.

Regularized models are sensitive to feature scale, so scaling is generally important when using them.

---

## 21. When Linear Regression Is a Good Choice

Linear Regression is especially useful when:

- The target is continuous
- Relationships are approximately linear
- Interpretability matters
- A transparent baseline is needed
- The dataset is not extremely high-dimensional
- Residual diagnostics are reasonably acceptable

It is often a strong first model even when more advanced algorithms will later be tested.

---

## 22. When Linear Regression May Not Be Enough

A basic linear model may be insufficient when:

- Relationships are strongly nonlinear
- Feature interactions are important
- There are complex thresholds or discontinuities
- Residual patterns show major model misspecification
- Outliers dominate the fit
- Predictive performance requires more flexible models

Possible alternatives include:

- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Other nonlinear models

---

## 23. Common Mistakes

### Mistake 1: Assuming the target must be normally distributed

It does not.

Residual assumptions are more relevant to classical linear-model inference.

### Mistake 2: Preprocessing the full dataset before splitting

This can introduce data leakage.

Fit preprocessing steps using training data only.

### Mistake 3: Scaling inference data differently from training data

Training and prediction must use the same preprocessing logic.

A Pipeline is one of the safest solutions.

### Mistake 4: Removing every feature with high VIF automatically

VIF is a diagnostic, not an automatic deletion rule.

### Mistake 5: Removing every outlier automatically

Outliers should be investigated before modification.

### Mistake 6: Judging the model only by R²

Use multiple metrics and residual diagnostics.

### Mistake 7: Interpreting coefficients as causal effects

Regression association does not automatically establish causality.

---

## 24. Interview Questions

### Q1. What is the difference between regression and classification?

Regression predicts continuous numerical values, while classification predicts discrete classes or categories.

### Q2. What does a Linear Regression coefficient mean?

It represents the expected change in the predicted target for a one-unit increase in that feature, holding the other predictors constant.

### Q3. What does the intercept represent?

It is the predicted target value when all features are zero, although that scenario may not always have a meaningful real-world interpretation.

### Q4. What is a residual?

A residual is the difference between an observed target value and the model's predicted value.

### Q5. What does Ordinary Least Squares minimize?

It minimizes the sum of squared residuals.

### Q6. Does Linear Regression require the target variable to be normally distributed?

No. Normality assumptions are generally associated with the residuals for classical statistical inference, not with requiring the raw target itself to be normal.

### Q7. Why is multicollinearity a problem?

It can make coefficient estimates unstable and difficult to interpret.

### Q8. What is the difference between MAE and RMSE?

Both measure prediction error, but RMSE penalizes large errors more strongly because it is derived from squared errors.

### Q9. What does R² measure?

It measures the proportion of target variance explained by the model relative to a mean baseline.

### Q10. Why use cross-validation?

Cross-validation provides a more reliable estimate of generalization by evaluating the model across multiple data splits.

### Q11. Does ordinary Linear Regression require feature scaling?

Not for valid predictions. However, scaling becomes important for regularized regression and many distance- or margin-based algorithms.

### Q12. What is data leakage?

Data leakage occurs when information unavailable at real prediction time improperly influences model training or preprocessing, producing overly optimistic evaluation results.

### Q13. Ridge vs. Lasso — what is the main difference?

Ridge uses an L2 penalty and generally shrinks coefficients, while Lasso uses an L1 penalty and can shrink some coefficients exactly to zero.

### Q14. When would you choose Linear Regression as a baseline?

When the target is continuous and a transparent, interpretable benchmark is useful before testing more complex models.

---

## 25. Connection to This Project

The practical implementation of the concepts in this tutorial is available in:

```text
../notebooks/boston_housing_linear_regression.ipynb
```

The notebook includes:

- Data validation
- Missing-value analysis
- Exploratory data analysis
- Pearson correlation analysis
- IQR-based outlier diagnostics
- Median imputation
- Linear Regression pipeline
- VIF diagnostics
- Test-set evaluation
- Residual analysis
- Q-Q plot
- Coefficient inspection
- 5-fold cross-validation
- Example prediction

---

## 26. Key Takeaways

Linear Regression is simple in form but requires careful reasoning.

A strong regression workflow is not just:

```text
fit → predict → R²
```

A better workflow is:

```text
Understand the problem
        ↓
Validate the data
        ↓
Explore relationships
        ↓
Prevent leakage
        ↓
Build a reproducible pipeline
        ↓
Train the model
        ↓
Evaluate with multiple metrics
        ↓
Inspect residuals
        ↓
Check multicollinearity
        ↓
Cross-validate
        ↓
Interpret results and limitations
```

That workflow is the main focus of the Boston Housing Linear Regression project.
