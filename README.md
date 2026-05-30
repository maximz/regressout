# regressout

[![](https://img.shields.io/pypi/v/regressout.svg)](https://pypi.python.org/pypi/regressout)
[![CI](https://github.com/maximz/regressout/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/maximz/regressout/actions/workflows/ci.yaml)
[![](https://img.shields.io/badge/docs-here-blue.svg)](https://regressout.maximz.com)
[![](https://img.shields.io/github/stars/maximz/regressout?style=social)](https://github.com/maximz/regressout)

`regressout` removes the linear effect of observed covariates from a feature
matrix. It provides `RegressOutCovariates`, a scikit-learn-style estimator
that residualizes each feature column against a covariate matrix.

## Why it exists

Some modeling workflows need features with variation explained by known
covariates removed first. For example, a feature matrix may need to be adjusted
for observed variables such as age, sex, ethnicity, batch, site, or other
metadata before downstream analysis. This package fits those adjustments and
returns the residual feature matrix.

## How it works

`RegressOutCovariates` uses scikit-learn naming, but with domain-specific
meaning:

- `X` is the covariate or observation matrix: the variables to regress out.
- `y` is the feature matrix to residualize.

On `fit(X=covariates, y=features)`, it fits one
`sklearn.linear_model.LinearRegression` model per feature column:

```text
feature_j ~ covariates
```

On `predict(X=covariates, y=features)`, it predicts the covariate contribution
for each feature and returns:

```text
feature_j - predicted_feature_j
```

If `y` is a pandas `DataFrame`, the returned residuals are also a `DataFrame`
with the same index and columns. Otherwise, residuals are returned as a NumPy
array.

## Installation

```bash
pip install regressout
```

For local development from this repository:

```bash
pip install -r requirements_dev.txt
pip install -e .
```

The runtime dependencies declared by the package are `numpy`, `pandas`, and
`scikit-learn`; Python 3.8 or newer is required.

## Usage

```python
import pandas as pd
from regressout import RegressOutCovariates

covariates = pd.DataFrame(
    {
        "age": [25, 49, 60, 50],
        "sex_M": [1, 0, 1, 0],
    },
    index=["sample1", "sample2", "sample3", "sample4"],
)

features = pd.DataFrame(
    {
        "feat1": [1.2, 2.5, 2.9, 3.1],
        "feat2": [0.4, 0.7, 1.4, 1.6],
    },
    index=covariates.index,
)

residualizer = RegressOutCovariates()
residualizer.fit(X=covariates, y=features)

residualized_features = residualizer.predict(X=covariates, y=features)
```

When covariates need preprocessing, put the preprocessing steps before
`RegressOutCovariates` in a scikit-learn pipeline. The tests show this pattern
with categorical encoding, column matching, scaling, and then residualization.

## Important behavior and limitations

- Covariates must already be numeric when they reach `RegressOutCovariates`.
  Encode categorical variables, impute missing values, or scale covariates in
  earlier pipeline steps as needed.
- The estimator performs independent linear regression for each feature column;
  it does not model nonlinear effects unless you add nonlinear covariate
  features before fitting.
- When fitted with pandas DataFrames, it validates row indexes and column order
  on later predictions where that metadata is available.
- The number of rows in `X` and `y` must match. The number and order of
  covariate and feature columns must match what was seen during `fit`.
- Unlike a standard scikit-learn estimator, both `fit` and `predict` take two
  arguments (`predict(X=covariates, y=features)`); a single-argument
  `predict(X)` call will not work, and the class is a predictor rather than a
  `transform`-style transformer.

## Development

```bash
make test
make lint
make docs
```

The package is MIT licensed.
