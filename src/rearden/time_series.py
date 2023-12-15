"""Time-series analysis tools."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

from .decorators import DataSplitterDecorators
from .preprocessings import (
    DataSplitter,
    RandomStateInstance,
    TrainTest,
    TrainValidTest,
)
from .vizualizations import save_plot_in_dir


class TimeSeriesFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Implements feature engineering for time series DataFrame.

    Attributes:
        col_name: Name of the column storing time-series.
        max_lag: Maximum lag to consider when generating features.
        rolling_mean_order: Rolling mean order to consider when generating features.
    """

    def __init__(
        self,
        col_name: str,
        max_lag: int = 1,
        rolling_mean_order: int = 1,
    ) -> None:
        """Constructor for TimeSeriesFeaturesExtractor class."""
        self.col_name = col_name
        self.max_lag = max_lag
        self.rolling_mean_order = rolling_mean_order

    def fit(self, X: pd.DataFrame, y=None) -> None:
        """Returns the object itself."""
        return self

    def _compute_lags(self, data: pd.DataFrame) -> pd.DataFrame:
        """Computes time lags and adds features to data."""
        for lag in range(1, self.max_lag + 1):
            data[f"lag_{lag}"] = data[self.col_name].shift(lag)

        return data

    def _compute_rolling_mean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Computes rolling mean and adds a feature to data."""
        data[f"rolling_mean_{self.rolling_mean_order}"] = (
            data[self.col_name].shift().rolling(self.rolling_mean_order).mean()
        )

        return data

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Adding new features to data."""
        # Adding time variables
        X["quarter"] = X.index.quarter
        X["month"] = X.index.month
        X["day"] = X.index.day
        X["hour"] = X.index.hour

        # Adding lags
        X = self._compute_lags(data=X)

        # Adding moving average
        X = self._compute_rolling_mean(data=X)

        return X


class TimeSeriesSplitter(DataSplitter):
    """Class implementing DataFrame split into sets for time-series.

    Attributes:
        set_shares: Tuple of shares in accordance with which data is to be split.
        random_seed: Number referring to random seed.
    """

    def __init__(
        self,
        set_shares: tuple[int],
        random_seed: Union[int, RandomStateInstance, None] = None,
    ) -> None:
        """Initializes instance based on data passed and split settings.

        Args:
            set_shares (tuple[int]): Includes a sequence of required set shares.
                Can refer to either `(train_share, valid_share, test_share)`
                or `(train_share, test_share)`.
            random_seed (Union[int, RandomStateInstance, None], optional): Stores a random seed.
                Defaults to None.
        """
        super().__init__(set_shares=set_shares, random_seed=random_seed)

    @DataSplitterDecorators.check_proportions(attr="set_shares")
    def _df_train_test_split_by_target(self, data, target) -> TrainTest:
        """Splits the features DataFrame and target vector into training and test sets."""
        # Recovering share of test set
        _, test_share = self.set_shares

        # Splitting the data into training/test sets
        training_set, test_set = train_test_split(
            data, shuffle=False, test_size=test_share
        )

        # Dropping nans due to lags
        training_set = training_set.dropna()

        # Separating features from target
        features_train, target_train = self._df_sep_by_target(
            data=training_set, target=target
        )

        features_test, target_test = self._df_sep_by_target(
            data=test_set, target=target
        )

        return features_train, target_train, features_test, target_test

    @DataSplitterDecorators.check_proportions(attr="set_shares")
    def _df_train_valid_test_split_by_target(
        self, data, target
    ) -> TrainValidTest:
        """Splits the features matrix and target vector into training, validation and test sets."""
        # Recovering shares of sets
        _, valid_share, test_share = self.set_shares

        # Determining cutoffs for splits
        split_1 = np.sum([valid_share, test_share])
        split_2 = test_share / split_1

        # Splitting the data into training/validation/test sets
        training_set, valid_test_set = train_test_split(
            data, shuffle=False, test_size=split_1
        )
        valid_set, test_set = train_test_split(
            valid_test_set, shuffle=False, test_size=split_2
        )

        # Dropping nans due to lags
        training_set = training_set.dropna()

        # Separating features from target
        features_train, target_train = self._df_sep_by_target(
            data=training_set, target=target
        )

        features_valid, target_valid = self._df_sep_by_target(
            data=valid_set, target=target
        )

        features_test, target_test = self._df_sep_by_target(
            data=test_set, target=target
        )

        return (
            features_train,
            target_train,
            features_valid,
            target_valid,
            features_test,
            target_test,
        )


class TimeSeriesPlotter:
    """Class for plotting time-series data."""

    def __init__(self) -> None:
        """Constructor for TimeSeriesPlotter class."""

    def _check_monotonicity(self, data: pd.DataFrame) -> None:
        """Time-series data monotonicity checker."""
        if data.index.is_monotonic_increasing is False:
            raise RuntimeError("Inconsistent dates")

    def resample_data(
        self, data: pd.DataFrame, periodicity: str = "1H"
    ) -> pd.DataFrame:
        """Resamples data according to periodicity."""
        # Checking the consistency of dates in data
        self._check_monotonicity(data=data)

        # Resampling data
        data_resampled = data.resample(periodicity).sum()

        return data_resampled

    def plot_decomposed(
        self,
        data: pd.DataFrame,
        periodicity: str = "1H",
        period: Optional[tuple[str, str]] = None,
        figure_dims: Optional[tuple[int, int]] = (8, 10),
        ylabel_name: str = "ylabel_name",
        save_fig: bool = False,
    ) -> Any:
        """Plots a decomposed time-series."""
        self._check_monotonicity(data=data)

        # Checking monotonicity
        data_resampled = self.resample_data(data=data, periodicity=periodicity)

        # Decomposing a time-series
        decomposed = seasonal_decompose(data_resampled)

        # Making a plot
        plt.figure(figsize=figure_dims)

        if period is not None:
            period_start, period_end = period

            trend = decomposed.trend[period_start:period_end]
            seasonal = decomposed.seasonal[period_start:period_end]
            resid = decomposed.resid[period_start:period_end]
        else:
            trend = decomposed.trend
            seasonal = decomposed.seasonal
            resid = decomposed.resid

        # Trend component
        plt.subplot(311)
        trend_plot = sns.lineplot(data=trend, ax=plt.gca())
        trend_plot.set(title="Trend", xlabel="Time period", ylabel=ylabel_name)
        plt.xticks(rotation=45)

        # Seasonal component
        plt.subplot(312)
        seasonal_plot = sns.lineplot(data=seasonal, ax=plt.gca())
        seasonal_plot.set(
            title="Seasonality", xlabel="Time period", ylabel=ylabel_name
        )
        plt.xticks(rotation=45)

        # Residual component
        plt.subplot(313)
        residual_plot = sns.lineplot(data=resid, ax=plt.gca())
        residual_plot.set(
            title="Residual", xlabel="Time period", ylabel=ylabel_name
        )
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Saving the figure
        if save_fig:
            save_plot_in_dir(file_name="ts_decomposed.png")

        plt.show()

    def plot_time_series(
        self,
        data: pd.DataFrame,
        periodicity: str = "1H",
        resample: bool = True,
        period: Optional[tuple[str, str]] = None,
        figure_dims: Optional[tuple[int, int]] = None,
        title_name: str = "title_name",
        ylabel_name: str = "ylabel_name",
        save_fig: bool = False,
    ) -> Any:
        """Plots a time-series."""
        self._check_monotonicity(data=data)

        data_resampled = data

        if resample:
            data_resampled = self.resample_data(
                data=data, periodicity=periodicity
            )

        plt.figure(figsize=figure_dims)

        data_to_plot = data_resampled

        if period is not None:
            period_start, period_end = period
            data_to_plot = data_resampled[period_start:period_end]

        # Plotting time series
        full_data_plot = sns.lineplot(data=data_to_plot, ax=plt.gca())
        full_data_plot.set(
            title=title_name, xlabel="Time period", ylabel=ylabel_name
        )
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_fig:
            save_plot_in_dir(file_name="ts_plot.png")

        plt.show()
