# Boston Housing Price Prediction with Linear Regression

An end-to-end machine learning project exploring the historical Boston Housing dataset and building an interpretable Linear Regression baseline for housing value prediction.

The project focuses not only on predictive performance, but also on the reasoning behind the machine learning workflow — including data validation, exploratory analysis, missing-value handling, multicollinearity, residual diagnostics, cross-validation, and model interpretation.

---

## Learning Resources

This repository also includes focused learning materials that explain the theory and statistics behind the project:

- [Linear Regression Tutorial](docs/linear_regression_tutorial.md)
- [Statistics for Regression](docs/statistics_for_regression.md)
- [Jupyter Notebook](notebooks/boston_housing_linear_regression.ipynb)

These resources connect the theoretical concepts directly to the implementation and diagnostics used in the Boston Housing project.

---

## 1. Project Overview

The goal of this project is to predict `MEDV`, the median value of owner-occupied homes in Boston-area towns, using 13 input features.

Since the target is numerical and continuous, this is a supervised regression problem.

Linear Regression is used as the baseline model because it provides a simple and interpretable starting point for understanding the relationship between housing characteristics and the target variable.

---

## 2. Business / Problem Statement

The objective is to build a transparent regression baseline that can answer three practical questions:

1. Which variables show the strongest relationships with housing values?
2. How accurately can a Linear Regression model predict unseen observations?
3. Do the residuals and model diagnostics support the assumptions behind a linear modeling approach?

Rather than treating prediction accuracy as the only goal, this project also examines the statistical behavior and limitations of the model.

---

## 3. Dataset

The dataset contains:

- **506 observations**
- **13 input features**
- **1 target variable: `MEDV`**

The records represent Boston-area suburbs or towns and originate from historical data collected around 1970.

During the initial data-quality assessment:

- No duplicate rows were found.
- Several features contained missing values.
- `MEDV` contained no missing values.
- A total of **120 missing values** were identified across the predictor columns.

Missing values were observed in:

- `CRIM`
- `ZN`
- `INDUS`
- `CHAS`
- `AGE`
- `LSTAT`

Each of these columns contained 20 missing observations.

> **Responsible-use note**
>
> This is a historical educational dataset. One of the original features (`B`) is race-derived, and the observations come from historical 1970-era data. This project is intended for machine learning education and should not be treated as a modern housing valuation, lending, or policy model.

---

## 4. Features

| Feature | Description |
|---|---|
| `CRIM` | Per-capita crime rate by town |
| `ZN` | Proportion of residential land zoned for large lots |
| `INDUS` | Proportion of non-retail business acres per town |
| `CHAS` | Charles River dummy variable |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to employment centers |
| `RAD` | Accessibility index to radial highways |
| `TAX` | Property-tax rate |
| `PTRATIO` | Pupil-teacher ratio |
| `B` | Historical race-derived feature from the original dataset |
| `LSTAT` | Percentage of lower-status population |
| `MEDV` | Median value of owner-occupied homes in $1000s |

---

## 5. Data Preprocessing

The preprocessing workflow was designed to reduce data leakage and keep the baseline model easy to interpret.

The dataset was first split into training and test sets. Missing values were then handled inside a Scikit-learn `Pipeline` using:

```python
SimpleImputer(strategy="median")
```

The imputer was therefore fitted only on the training data during model training.

Feature scaling was intentionally not applied to the baseline Ordinary Least Squares Linear Regression model because scaling is not required for its predictions, and keeping the original units makes coefficient interpretation more straightforward.

---

## 6. Exploratory Data Analysis (EDA)

The exploratory analysis included:

- Dataset structure and data types
- Missing-value inspection
- Duplicate detection
- Target distribution
- Boxplot analysis
- Pearson correlation analysis
- Feature-vs-target scatter plots
- IQR-based outlier diagnostics

The analysis showed that:

- `RM` has a strong positive linear relationship with `MEDV`.
- `LSTAT` has a strong negative linear relationship with `MEDV`.
- The target contains multiple observations at `MEDV = 50`, so the upper end of the target distribution should be interpreted with caution.

Pearson correlation was used because the baseline model focuses on linear relationships.

---

## 7. Statistical Analysis

The statistical analysis focused on concepts directly relevant to regression modeling:

- Mean and median
- Variance and standard deviation
- Pearson correlation
- Covariance
- Distribution shape
- Outlier detection
- Residual diagnostics
- Normality assessment
- Multicollinearity
- Variance Inflation Factor (VIF)

VIF was used as a diagnostic rather than as an automatic feature-removal rule.

Some predictors, particularly variables such as `RAD` and `TAX`, showed evidence of multicollinearity. These relationships were documented rather than automatically removed so the baseline model could remain transparent and reproducible.

---

## 8. Feature Engineering

This project intentionally keeps feature engineering limited in order to establish a clear and interpretable baseline.

Outliers were identified using the IQR method, but observations were not automatically clipped or removed.

This preserves the original structure of the historical dataset and avoids introducing arbitrary transformations before baseline performance is established.

Potential future improvements include:

- Polynomial features
- Interaction terms
- Feature selection
- Robust transformations
- Regularization

---

## 9. Model

The baseline model is **Linear Regression**.

Linear Regression models the target as a linear combination of the input features:

```text
Prediction = Intercept + Coefficient₁ × Feature₁ + ... + Coefficientₙ × Featureₙ
```

The coefficients are estimated using Ordinary Least Squares (OLS), which minimizes the sum of squared residuals between observed and predicted values.

The fitted coefficients were extracted directly from the trained model rather than hard-coded.

---

## 10. Model Training

The model was implemented using a Scikit-learn `Pipeline` containing:

1. Median imputation
2. Linear Regression

The dataset was divided into training and testing subsets using a fixed random state for reproducibility.

A 5-fold shuffled cross-validation procedure was also used to evaluate whether model performance remained reasonably stable across multiple train-validation splits.

---

## 11. Evaluation Metrics

The model was evaluated using:

- **MAE — Mean Absolute Error**
- **MSE — Mean Squared Error**
- **RMSE — Root Mean Squared Error**
- **R² — Coefficient of Determination**
- **Adjusted R²**

Lower values are better for MAE, MSE, and RMSE.

Higher values are generally better for R² and Adjusted R².

Model evaluation also included:

- Actual vs. predicted values
- Residual plot
- Residual distribution
- Q-Q plot
- Cross-validation performance

---

## 12. Results

### Test Set Performance

| Metric | Result |
|---|---:|
| MAE | ~3.15 |
| MSE | ~24.98 |
| RMSE | ~5.00 |
| R² | ~0.659 |
| Adjusted R² | ~0.609 |

The model explains approximately 66% of the variance in the held-out test target.

### 5-Fold Cross-Validation

| Metric | Mean Result |
|---|---:|
| R² | ~0.708 |
| MAE | ~3.415 |
| RMSE | ~4.912 |

Cross-validation provides a more robust estimate of generalization performance than relying only on a single train/test split.

---

## 13. Conclusion

Linear Regression provides a useful and interpretable baseline for the Boston Housing dataset.

The model captures a substantial portion of the variation in housing values while remaining straightforward to inspect and explain.

This project also demonstrates why predictive metrics alone are not sufficient for evaluating a regression model. Residual behavior, multicollinearity, data quality, historical context, and model assumptions must also be considered.

The baseline leaves room for improvement, particularly through regularization, nonlinear relationships, and more flexible regression algorithms.

---

## 14. Future Improvements

Potential next steps include:

- Ridge Regression
- Lasso Regression
- Elastic Net
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Hyperparameter tuning
- Feature selection
- More detailed residual diagnostics
- Comparison of multiple regression algorithms

These extensions can help determine whether more flexible models provide meaningful improvements over the linear baseline.

---

## 15. Technologies & Libraries

### Language

- Python

### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Statsmodels

### Tools

- Jupyter Notebook
- Git
- GitHub
- Kaggle

---

## 16. Kaggle Notebook

A Kaggle version of this project will be published for interactive execution.

**Kaggle Notebook:** Coming soon.

---

## Project Structure

```text
boston-housing-linear-regression/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   └── HousingData.csv
│
├── notebooks/
│   └── boston_housing_linear_regression.ipynb
│
└── docs/
    ├── linear_regression_tutorial.md
    └── statistics_for_regression.md
```

---

## What I Learned

This project helped reinforce several important machine learning practices:

- Building reproducible preprocessing pipelines
- Preventing data leakage
- Understanding when feature scaling is necessary
- Interpreting regression coefficients
- Evaluating regression models with multiple metrics
- Using cross-validation instead of relying only on one train/test split
- Diagnosing multicollinearity
- Examining residual behavior
- Separating predictive performance from statistical assumptions
- Documenting machine learning work for reproducibility and portfolio presentation

---

## Repository Contents

The repository contains:

- A reproducible Jupyter Notebook
- The dataset used in the project
- Project dependencies
- A Linear Regression tutorial
- A statistics-for-regression guide
- Full project documentation

You can start with the [Jupyter Notebook](notebooks/boston_housing_linear_regression.ipynb) or review the [Linear Regression Tutorial](docs/linear_regression_tutorial.md) first.
