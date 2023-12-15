"""Preprocessing tools."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .decorators import DataSplitterDecorators

# Type aliases for type signatures
FeaturesTarget = tuple[pd.DataFrame, pd.Series]
TrainTest = tuple[FeaturesTarget, FeaturesTarget]
TrainValidTest = tuple[FeaturesTarget, FeaturesTarget, FeaturesTarget]
RandomStateInstance = np.random.RandomState


class DataSplitter:
    """Class implementing DataFrame split into sets.

    Based on the number of shares provided, splits the
    passed data either into training/validation/test sets
    or training/test sets. Additionally, separates features
    from target in the DataFrame containing both.

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
        self.set_shares = set_shares
        self.random_seed = random_seed
        self.features_train_ = None
        self.features_valid_ = None
        self.features_test_ = None
        self.target_train_ = None
        self.target_valid_ = None
        self.target_test_ = None
        self.data_size_ = 0

    def _df_sep_by_target(
        self, data: pd.DataFrame, target: str
    ) -> FeaturesTarget:
        """Separates DataFrame into target vector and DataFrame with features."""
        # Separating features from target
        features = data.drop([target], axis=1)
        # Singling out the target vector
        target = data[target]

        return features, target

    @DataSplitterDecorators.check_proportions(attr="set_shares")
    def _df_train_test_split_by_target(
        self, data: pd.DataFrame, target: str
    ) -> TrainTest:
        """Splits the features DataFrame and target vector into training and test sets."""
        # Separating features from target
        features, target = self._df_sep_by_target(data=data, target=target)
        # Recovering share of test set
        _, test_share = self.set_shares
        # Splitting features and target into training and test sets
        (
            features_train,
            features_test,
            target_train,
            target_test,
        ) = train_test_split(
            features,
            target,
            test_size=test_share,
            random_state=self.random_seed,
        )

        return features_train, target_train, features_test, target_test

    @DataSplitterDecorators.check_proportions(attr="set_shares")
    def _df_train_valid_test_split_by_target(
        self, data: pd.DataFrame, target: str
    ) -> TrainValidTest:
        """Splits the features matrix and target vector into training, validation and test sets."""
        # Separating features from target
        features, target = self._df_sep_by_target(data=data, target=target)
        # Recovering shares of validation and test sets
        _, valid_share, test_share = self.set_shares
        # Determining split cutoff points consistent with defined shares
        split_1 = np.sum([valid_share, test_share])
        split_2 = test_share / split_1
        # Splitting features and target into training set and a combination of validation/test sets
        (
            features_train,
            features_valid_test,
            target_train,
            target_valid_test,
        ) = train_test_split(
            features,
            target,
            test_size=split_1,
            random_state=self.random_seed,
        )
        # Separating validation set from test set
        (
            features_valid,
            features_test,
            target_valid,
            target_test,
        ) = train_test_split(
            features_valid_test,
            target_valid_test,
            test_size=split_2,
            random_state=self.random_seed,
        )

        return (
            features_train,
            target_train,
            features_valid,
            target_valid,
            features_test,
            target_test,
        )

    @DataSplitterDecorators.check_dimensions(attr="set_shares")
    def split_data(
        self, data: pd.DataFrame, target: str
    ) -> Union[TrainTest, TrainValidTest]:
        """Splits the data into sets according to set shares passed."""
        # Recovering the number of objects in the DataFrame
        self.data_size_ = data.shape[0]
        # Checking if we can split into train/test given set shares passed
        try:
            (
                features_train,
                target_train,
                features_test,
                target_test,
            ) = self._df_train_test_split_by_target(data=data, target=target)
        # In case of inability to unpack the tuple with set shares, another function executed
        except ValueError:
            (
                features_train,
                target_train,
                features_valid,
                target_valid,
                features_test,
                target_test,
            ) = self._df_train_valid_test_split_by_target(
                data=data, target=target
            )
            # Saving the splits into object attributes
            self.features_train_ = deepcopy(features_train)
            self.target_train_ = target_train
            self.features_test_ = deepcopy(features_test)
            self.target_test_ = target_test
            self.features_valid_ = deepcopy(features_valid)
            self.target_valid_ = target_valid
            # Adding a new attribute for split data
            self.data_splitted_ = True

            return (
                features_train,
                target_train,
                features_valid,
                target_valid,
                features_test,
                target_test,
            )
        # If the data was split into train/test, splits are saved into attributes
        self.features_train_ = deepcopy(features_train)
        self.target_train_ = target_train
        self.features_test_ = deepcopy(features_test)
        self.target_test_ = target_test
        # Adding a new attribute for split data
        self.data_splitted_ = True

        return (
            features_train,
            target_train,
            features_test,
            target_test,
        )

    @property
    @DataSplitterDecorators.check_split(
        attr="data_splitted_", func_hint="split_data"
    )
    def subset_info_(self) -> pd.DataFrame:
        """Displays table with sets observation numbers and sets shares."""
        # Computing the shares of sets
        features_train_share = self.features_train_.shape[0] / self.data_size_
        features_train_share = np.round(features_train_share, 2)
        features_test_share = self.features_test_.shape[0] / self.data_size_
        features_test_share = np.round(features_test_share, 2)
        # Checking if the splits include validation set
        try:
            features_valid_share = (
                self.features_valid_.shape[0] / self.data_size_
            )
            features_valid_share = np.round(features_valid_share, 2)
        # In case there was no validation split, info about train/test is used
        except AttributeError:
            index = ["train", "test"]
            data = {
                "obs_num": [
                    self.features_train_.shape[0],
                    self.features_test_.shape[0],
                ],
                "set_shares": [features_train_share, features_test_share],
            }
        # In other case, we have information about all three sets
        else:
            index = ["train", "valid", "test"]
            data = {
                "obs_num": [
                    self.features_train_.shape[0],
                    self.features_valid_.shape[0],
                    self.features_test_.shape[0],
                ],
                "set_shares": [
                    features_train_share,
                    features_valid_share,
                    features_test_share,
                ],
            }
        # Creating the output DataFrame
        set_shares_info = pd.DataFrame(data=data, index=index)
        # Fixing output format of columns
        set_shares_info["obs_num"] = set_shares_info["obs_num"].map(
            "{:,}".format
        )
        set_shares_info["set_shares"] = set_shares_info["set_shares"].map(
            "{:,}".format
        )

        return set_shares_info


def identify_missing_values(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Performs missing values computation.

    Computes a number and share of missing values
    in DataFrame columns which have NaN-values present
    and displays data types of such columns.

    Args:
        data (pd.DataFrame): DataFrame which needs to
            be checked for missing values.

    Returns:
        Optional[pd.DataFrame]: DataFrame with column names, number of
        missing values and shares of NaN-values in such columns.
        In case there are not missing values present, None is returned.
        Additionally, columns data type is shown.
    """
    # Verifying the presence of missing values
    miss_vals_num = data.isnull().sum()[data.isnull().sum() > 0]
    if miss_vals_num.empty:
        return None

    # Creating a table with numbers of missing values
    cols = {"missing_count": miss_vals_num.values}
    nans_df = pd.DataFrame(data=cols, index=miss_vals_num.index).sort_values(
        by="missing_count", ascending=False
    )

    # Adding shares of missing values
    nans_df["missing_fraction"] = nans_df["missing_count"] / data.shape[0]
    nans_df["missing_fraction"] = nans_df["missing_fraction"].round(4)

    # Adding data types
    nans_df["dtype"] = data[nans_df.index].dtypes
    nans_df = nans_df[["dtype", "missing_count", "missing_fraction"]]

    return nans_df
