"""Regress Out Covariates."""

from __future__ import annotations

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.1"

import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array

logger = logging.getLogger(__name__)
# Set default logging handler to avoid "No handler found" warnings.
logger.addHandler(logging.NullHandler())


class RegressOutCovariates(ClassifierMixin, BaseEstimator):
    """
    Regress out a set of variables from the feature matrix.

    This is a preprocessing transformation of the feature matrix, but works as a sklearn predictor.
    Under the hood: a bunch of linear regression fits.

    Suppose you want to train a model to predict a target variable Y, given a set of features X.
    But first you want to regress out the impact of another set of features Obs from X.
    Fit will fit a regression function `Xi ~ Obs` for each column Xi in X.
    Predict will regress out Obs from X, returning Xi - Xi_hat = Xi - f_i(Obs) for each column Xi of X.

    But the parameter names are different (to follow sklearn convention):
    Fit and predict parameters `X` is where you pass in Obs matrix, and `Y` is where you pass in X feature matrix.

    If you want to standardize the obs matrix before fitting regression to each column of the feature matrix, wrap RegressOutTransformer in a pipeline with StandardScaler preceding it. (Similarly make any dummy variables in obs matrix first.)
    """

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ) -> RegressOutCovariates:
        """X is obs matrix, y is the feature matrix"""
        if type(X) == pd.DataFrame:
            # set column order if available
            self.obs_feature_order_ = X.columns
        else:
            # no column order available - will skip this validation
            self.obs_feature_order_ = None

        if type(y) == pd.DataFrame:
            # set column order if available
            self.feature_matrix_feature_order_ = y.columns
        else:
            # no column order available - will skip this validation
            self.feature_matrix_feature_order_ = None

        # Validate that row indices match, if available
        if type(X) == pd.DataFrame and type(y) == pd.DataFrame:
            if not np.array_equal(X.index, y.index):
                raise ValueError(
                    "The index of the feature matrix DataFrame must match the index of the DataFrame with variables to be regressed out."
                )

        # Input validation, and convert to numpy arrays
        obs_vals = check_array(X, accept_sparse=True)
        feature_matrix_vals = check_array(y, accept_sparse=True)

        self.obs_vals_n_features_ = obs_vals.shape[1]

        if obs_vals.shape[0] != feature_matrix_vals.shape[0]:
            raise ValueError(
                "The number of rows in the feature matrix must match the number of rows in the variables to be regressed out."
            )

        self.transformations_ = []
        for colix in range(feature_matrix_vals.shape[1]):
            self.transformations_.append(
                self._fit_regression(
                    feature_matrix_single_column_vector=feature_matrix_vals[:, colix],
                    obs_values_matrix=obs_vals,
                )
            )

        list_of_regressed_out_variables = (
            f" ({list(self.obs_feature_order_)})"
            if self.obs_feature_order_ is not None
            else ""
        )
        logger.info(
            f"Fitted {len(self.transformations_)} transformations (one per column of feature matrix), regressing out {self.obs_vals_n_features_} variables{list_of_regressed_out_variables}."
        )

        return self

    @staticmethod
    def _fit_regression(
        feature_matrix_single_column_vector: np.ndarray, obs_values_matrix: np.ndarray
    ) -> linear_model.LinearRegression:
        """Fit regression for a single column:
        feature_matrix_single_column_vector is a vector, obs_values_matrix is a matrix
        """
        return linear_model.LinearRegression().fit(
            X=obs_values_matrix, y=feature_matrix_single_column_vector
        )

    @staticmethod
    def _transform_regression(
        feature_matrix_single_column_vector: np.ndarray,
        obs_values_matrix: np.ndarray,
        fitted_regression: linear_model.LinearRegression,
    ) -> np.ndarray:
        """Apply regression fitted for a single column.
        feature_matrix_single_column_vector is a vector, obs_values_matrix is a matrix.
        Returns residuals.
        """
        predicted_feature_matrix_values = fitted_regression.predict(obs_values_matrix)
        residuals = (
            feature_matrix_single_column_vector - predicted_feature_matrix_values
        )
        return residuals

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """X is obs, y is the feature matrix. Returns transformed feature matrix y."""
        # check if fit is performed prior to transform
        check_is_fitted(self, ["transformations_", "obs_vals_n_features_"])

        # Validate column orders if available
        if type(X) == pd.DataFrame and self.obs_feature_order_ is not None:
            if not np.array_equal(X.columns, self.obs_feature_order_):
                raise ValueError(
                    "The columns of the input obs DataFrame do not match the columns of the training obs DataFrame."
                )

        # Validate column orders if available
        if type(y) == pd.DataFrame and self.feature_matrix_feature_order_ is not None:
            if not np.array_equal(y.columns, self.feature_matrix_feature_order_):
                raise ValueError(
                    "The columns of the input feature matrix DataFrame do not match the columns of the training feature matrix DataFrame."
                )

        # Validate that row indices match, if available
        if type(X) == pd.DataFrame and type(y) == pd.DataFrame:
            if not np.array_equal(X.index, y.index):
                raise ValueError(
                    "The index of the feature matrix DataFrame must match the index of the DataFrame with variables to be regressed out."
                )

        # Input validation, and convert to numpy arrays
        obs_vals = check_array(X, accept_sparse=True)
        feature_matrix_vals = check_array(y, accept_sparse=True)

        # Check that the input is of the same shape as the one passed during fit.
        if feature_matrix_vals.shape[1] != len(self.transformations_):
            raise ValueError(
                "Number of columns of feature matrix input is different from what was seen in `fit`."
            )

        if obs_vals.shape[1] != self.obs_vals_n_features_:
            raise ValueError(
                "Number of columns of obs matrix input is different from what was seen in `fit`."
            )

        if obs_vals.shape[0] != feature_matrix_vals.shape[0]:
            raise ValueError(
                "The number of rows in the feature matrix must match the number of rows in the variables to be regressed out."
            )

        transformed_feature_matrix_vals = np.column_stack(
            [
                self._transform_regression(
                    feature_matrix_single_column_vector=feature_matrix_vals[:, colix],
                    obs_values_matrix=obs_vals,
                    fitted_regression=self.transformations_[colix],
                )
                for colix in range(feature_matrix_vals.shape[1])
            ]
        )

        if transformed_feature_matrix_vals.shape != feature_matrix_vals.shape:
            raise ValueError(
                f"The shape of the transformed feature matrix {transformed_feature_matrix_vals.shape} does not match the shape of the input feature matrix {feature_matrix_vals.shape}."
            )

        if type(y) == pd.DataFrame:
            # re-wrap as dataframe before returning
            return pd.DataFrame(
                transformed_feature_matrix_vals, columns=y.columns, index=y.index
            )

        # return as numpy array
        return transformed_feature_matrix_vals
