from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Specifying custom types
TrainValidTest = Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]
TrainTest = Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


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
        In case there are not missing values present, an according
        message is shown and nothing is returned. Additionally, columns
        data type is shown.
    """
    # Verifying missing values
    miss_vals_num = data.isnull().sum()[data.isnull().sum() > 0]
    if miss_vals_num.empty:
        print("Missing values are not found.")
        return
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


def preprocess_duplicates(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Computes and deletes all duplicates in the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame which needs to
            be checked for identical rows.
    """
    num_duplicates = data.duplicated().sum()
    if num_duplicates != 0:
        data.drop_duplicates(inplace=True)
        print(f"{num_duplicates:,} duplicates found and deleted.")
    else:
        print("No duplicates found.")


def prepare_sets(
    data: pd.DataFrame,
    target_name: str,
    train_share: float,
    test_share: float,
    valid_share: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Union[TrainValidTest, TrainTest]:
    """
    Conducts data split into sets.

    Depending on the value passed to `valid_share`,
    splits the data either into training, validation
    and test sets or into training and test sets.

    Args:
        data (pd.DataFrame): DataFrame which needs to be split into sets.
        target_name (str): Name of the target variable.
        train_share (float): Share of the training set.
        test_share (float): Share of the test set.
        valid_share (Optional[float], optional): Share of the validation set. Defaults to None.
        random_state (Optional[int], optional): Random seed. Defaults to None.

    Returns:
        Union[TrainValidTest, TrainTest]: Tuple of training,
        validation and test sets or Tuple of training and
        test sets.

    Raises:
        ValueError: Exception raised in case proportions
        of sets are inconsistent.
    """
    # Separating features from target
    features = data.drop([target_name], axis=1)
    target = data[target_name]

    # Data split into sets with a validation set
    if valid_share is not None:
        # Checking consistency of set shares
        if np.sum([train_share, valid_share, test_share]) != 1.0:
            raise ValueError("Incorrect sets proportions specified")

        # Determining cutoffs
        split_1 = 1 - train_share
        split_2 = test_share / np.sum([valid_share, test_share])

        # Splitting into training set and a combination of validation/test sets
        (
            features_train,
            features_valid_test,
            target_train,
            target_valid_test,
        ) = train_test_split(
            features, target, test_size=split_1, random_state=random_state
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
            random_state=random_state,
        )

        return (
            features_train,
            target_train,
            features_valid,
            target_valid,
            features_test,
            target_test,
        )

    # Data split into sets without a validation set
    if np.sum([train_share, test_share]) != 1.0:
        raise ValueError("Incorrect sets proportions specified")

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = train_test_split(
        features, target, test_size=test_share, random_state=random_state
    )

    return features_train, target_train, features_test, target_test
