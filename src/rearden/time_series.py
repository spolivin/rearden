from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

# Specifying custom types
TrainValidTest = Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]
TrainTest = Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


class FeaturesExtractor(BaseEstimator, TransformerMixin):
    """Implements feature engineering for time series DataFrame."""

    def __init__(
        self,
        max_lag: int = 1,
        rolling_mean_order: int = 1,
    ) -> None:
        """Constructor for FeaturesExtractor class."""
        self.max_lag = max_lag
        self.rolling_mean_order = rolling_mean_order

    def fit(self, x: pd.DataFrame, y=None) -> None:
        """Returns the object itself."""
        return self

    def transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Extracts features from DateTimeIndex in accordance
        with arguments passed to `__init__`.
        """
        X = x.copy()
        # Adding time variables
        X["quarter"] = X.index.quarter
        X["month"] = X.index.month
        X["day"] = X.index.day
        X["hour"] = X.index.hour
        # Adding lags
        for lag in range(1, self.max_lag + 1):
            X["lag_{}".format(lag)] = X["num_orders"].shift(lag)
        # Adding moving average
        X["rolling_mean_{}".format(self.rolling_mean_order)] = (
            X["num_orders"].shift().rolling(self.rolling_mean_order).mean()
        )

        return X


def prepare_ts(
    data: pd.DataFrame,
    target_name: str,
    train_share: float,
    test_share: float,
    valid_share: Optional[float] = None,
) -> Union[TrainValidTest, TrainTest]:
    """Conducts time series data split into sets.

    Depending on the value passed to `valid_share`,
    splits the data either into training, validation
    and test sets or into training and test sets.

    Args:
        data (pd.DataFrame): DataFrame which needs to be
            split into sets.
        target_name (str): Name of the target variable.
        train_share (float): Share of the training set.
        test_share (float): Share of the test set.
        valid_share (Optional[float], optional): Share of
            the validation set. Defaults to None.

    Returns:
        Union[TrainValidTest, TrainTest]: Tuple of training,
        validation and test sets or Tuple of training and
        test sets.

    Raises:
        ValueError: Exception raised in case proportions
        of sets are inconsistent.
    """
    # Data split into train/validation/test
    if valid_share is not None:
        # Checking consistency of set shares
        if np.sum([train_share, valid_share, test_share]) != 1.0:
            raise ValueError("Incorrect sets proportions specified")

        split_1 = 1 - train_share
        split_2 = test_share / np.sum([valid_share, test_share])

        training_set, valid_test_set = train_test_split(
            data, shuffle=False, test_size=split_1
        )
        valid_set, test_set = train_test_split(
            valid_test_set, shuffle=False, test_size=split_2
        )

        # Dropping nans due to lags
        training_set = training_set.dropna()

        features_train = training_set.drop([target_name], axis=1)
        target_train = training_set[target_name]

        features_valid = valid_set.drop([target_name], axis=1)
        target_valid = valid_set[target_name]

        features_test = test_set.drop([target_name], axis=1)
        target_test = test_set[target_name]

        return (
            features_train,
            target_train,
            features_valid,
            target_valid,
            features_test,
            target_test,
        )

    # Data split into training/test
    if np.sum([train_share, test_share]) != 1.0:
        raise ValueError("Incorrect sets proportions specified")

    # Data split
    training_set, test_set = train_test_split(
        data, shuffle=False, test_size=test_share
    )

    # Dropping nans due to lags
    training_set = training_set.dropna()

    # Separating features from target
    features_train = training_set.drop([target_name], axis=1)
    target_train = training_set[target_name]

    features_test = test_set.drop([target_name], axis=1)
    target_test = test_set[target_name]

    return features_train, target_train, features_test, target_test


def plot_time_series(
    data: pd.DataFrame,
    col: str,
    period_start: str,
    period_end: str,
    kind: Optional[str] = None,
) -> Any:
    """Plots time series.

    When specifying kind="decomposed", conducts seasonal decomposition
    of the data and outputs graphs of trend, seasonal and residual
    components.

    Args:
        data (pd.DataFrame): DataFrame with information
            about taxi orders.
        col (str): DataFrame column containing
            data about taxi orders.
        period_start (str): Time period start.
        period_end (str): Time period end.
        kind (Optional[str], optional): Boolean indicator
            of performing time-series decomposition.
    """
    # Plotting decomposed time series
    if kind == "decomposed":
        decomposed = seasonal_decompose(data)

        plt.figure(figsize=(8, 10))

        # Trend component
        plt.subplot(311)
        trend_plot = sns.lineplot(
            data=decomposed.trend[period_start:period_end], ax=plt.gca()
        )
        trend_plot.set(title="Trend", xlabel="Time period", ylabel="Orders")
        plt.xticks(rotation=45)

        # Seasonal component
        plt.subplot(312)
        seasonal_plot = sns.lineplot(
            data=decomposed.seasonal[period_start:period_end], ax=plt.gca()
        )
        seasonal_plot.set(
            title="Seasonality", xlabel="Time period", ylabel="Orders"
        )
        plt.xticks(rotation=45)

        # Residual component
        plt.subplot(313)
        residual_plot = sns.lineplot(
            data=decomposed.resid[period_start:period_end], ax=plt.gca()
        )
        residual_plot.set(
            title="Residual", xlabel="Time period", ylabel="Orders"
        )
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        return

    # Plotting time series
    full_data_plot = sns.lineplot(
        data=data[period_start:period_end],
        y=col,
        x=data[period_start:period_end].index,
        ax=plt.gca(),
    )
    full_data_plot.set(
        title="Taxi orders number", xlabel="Time period", ylabel="Orders"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
