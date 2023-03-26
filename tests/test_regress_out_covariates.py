import numpy as np
import pandas as pd
import pytest
import sklearn
from feature_engine.encoding import OneHotEncoder
from feature_engine.preprocessing import MatchVariables
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regressout import RegressOutCovariates


def test_sklearn_clonable():
    estimator = RegressOutCovariates()
    # Check that supports cloning with sklearn.base.clone
    estimator_clone = sklearn.base.clone(estimator)

    # not fitted yet
    assert not hasattr(estimator, "transformations_")
    assert not hasattr(estimator_clone, "transformations_")

    # pretend it is fitted
    estimator.transformations_ = []
    assert hasattr(estimator, "transformations_")

    # confirm clone is not fitted
    estimator_clone_2 = sklearn.base.clone(estimator)
    assert not hasattr(estimator_clone_2, "transformations_")


@pytest.fixture
def data():
    obs_train = pd.DataFrame(
        {
            "ethnicity": ["ethnicityA", "ethnicityB", "ethnicityA", "ethnicityC"],
            "age": [25, 49, 60, 50],
            "sex": ["M", "F", "M", "F"],
        },
        index=["train_sample1", "train_sample2", "train_sample3", "train_sample4"],
    )
    obs_test = pd.DataFrame(
        {
            "ethnicity": ["ethnicityB", "ethnicityA", "ethnicityC"],
            "age": [35, 59, 45],
            "sex": ["F", "M", "M"],
        },
        index=["test_sample1", "test_sample2", "test_sample3"],
    )
    n_features = 10
    feature_matrix_cols = [f"feat{n+1}" for n in range(n_features)]
    X_train = pd.DataFrame(
        np.random.randn(4, n_features),
        columns=feature_matrix_cols,
        index=["train_sample1", "train_sample2", "train_sample3", "train_sample4"],
    )
    X_test = pd.DataFrame(
        np.random.randn(3, n_features),
        columns=feature_matrix_cols,
        index=["test_sample1", "test_sample2", "test_sample3"],
    )

    return obs_train, obs_test, X_train, X_test


def test_regress_out(data):
    obs_train, obs_test, X_train, X_test = data

    # Make pipeline:
    # 1. create dummy variables from obs. for variables with only two categories (like sex), only use one dummy variable.
    # 2. confirm obs variables are in same order (puts in same order if they're not; throws error if any column missing; drops any test column not found in train)
    # 3. scale obs (wrapper keeps pandas dataframe structure)
    # 4. regress out this obs matrix from the feature matrix
    pipeline = make_pipeline(
        OneHotEncoder(
            variables=["ethnicity", "sex"], drop_last_binary=True, drop_last=False
        ),
        MatchVariables(missing_values="raise"),
        SklearnTransformerWrapper(StandardScaler()),
        RegressOutCovariates(),
    )

    pipeline.fit(obs_train, X_train)

    obs_train_transformed = pipeline[:-1].transform(X=obs_train)
    assert type(obs_train_transformed) == pd.DataFrame
    assert np.array_equal(obs_train_transformed.index, X_train.index)
    assert np.array_equal(
        obs_train_transformed.columns,
        [
            "age",
            "ethnicity_ethnicityA",
            "ethnicity_ethnicityB",
            "ethnicity_ethnicityC",
            "sex_M",
        ],
    )

    X_train_transformed = pipeline.predict(X=obs_train, y=X_train)
    assert X_train_transformed.shape == X_train.shape
    assert type(X_train_transformed) == pd.DataFrame
    assert np.array_equal(X_train_transformed.index, X_train.index)
    assert np.array_equal(X_train_transformed.columns, X_train.columns)

    obs_test_transformed = pipeline[:-1].transform(X=obs_test)
    assert type(obs_test_transformed) == pd.DataFrame
    assert np.array_equal(obs_test_transformed.index, X_test.index)
    assert np.array_equal(obs_test_transformed.columns, obs_train_transformed.columns)

    X_test_transformed = pipeline.predict(X=obs_test, y=X_test)
    assert X_test_transformed.shape == X_test.shape
    assert type(X_test_transformed) == pd.DataFrame
    assert np.array_equal(X_test_transformed.index, X_test.index)
    assert np.array_equal(X_test_transformed.columns, X_test.columns)

    # now run as numpy
    X_test_transformed_np = pipeline.predict(X=obs_test, y=X_test.values)
    assert type(X_test_transformed_np) == np.ndarray
    assert np.array_equal(X_test_transformed_np, X_test_transformed.values)

    # go column by column in feature matrix and do regression manually to verify values
    for colname in X_train.columns:
        fitted_regression = linear_model.LinearRegression().fit(
            obs_train_transformed.values, X_train[colname].values
        )
        train_residuals = X_train[colname].values - fitted_regression.predict(
            obs_train_transformed.values
        )
        test_residuals = X_test[colname].values - fitted_regression.predict(
            obs_test_transformed.values
        )
        assert np.allclose(train_residuals, X_train_transformed[colname].values)
        assert np.allclose(test_residuals, X_test_transformed[colname].values)
