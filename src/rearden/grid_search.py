from typing import (
    Any,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

from .decorators import exec_timer

# Specifying customs types
RandomStateInstance = np.random.mtrand.RandomState


class RandomizedHyperoptRegression(RandomizedSearchCV):
    """
    Wrapper for RandomizedSearchCV with custom defaults
    and additional functionality for running regressions.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Union[Sequence[Mapping], Mapping],
        train_dataset: Tuple[Any, Any],
        eval_dataset: Tuple[Any, Any],
        cv: Union[Iterable, int] = 2,
        random_state: Union[RandomStateInstance, int] = 12345,
        scoring: str = "neg_root_mean_squared_error",
        n_iter: int = 5,
        n_jobs: Optional[int] = None,
        return_train_score: bool = True,
    ) -> None:
        """Initializes an instance.

        Args:
            estimator (BaseEstimator): Sklearn model, Gradient Boosting model or
                Pipeline object.
            param_distributions (Union[Sequence[Mapping], Mapping]): Grid of
                hyperparameter names and their values to be varied.
            train_dataset (Tuple[Any, Any]): Tuple of training features and
                training target-vector.
            eval_dataset (Tuple[Any, Any]): Tuple of testing features and
                testing target-vector.
            cv (Union[Iterable, int], optional): Crossvalidator represented
                as an iterator or an integer. Defaults to 2.
            random_state (Union[RandomStateInstance, int], optional): Random seed
                represented as either RandomState instance or an integer.
                Defaults to 12345.
            scoring (str, optional): Scoring metric.
                Defaults to "neg_root_mean_squared_error".
            n_iter (int, optional): Number of hyperparameter combinations
                to consider. Defaults to 5.
            n_jobs (Optional[int], optional): Parallelization of computations.
                Defaults to None.
            return_train_score (bool, optional): Boolean indicating returning
                metric values scores computed on training set. Defaults to True.
        """
        super().__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            cv=cv,
            random_state=random_state,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_features, self.train_target = self.train_dataset
        self.eval_features, self.eval_target = self.eval_dataset

    @property
    def _model_name(self):
        """Retrieves last estimator name.

        The function which ends up behaving as a class property
        returns the name of the last estimator in the pipeline or
        the plain model name used.
        """
        try:
            pipeline_steps = self.estimator.named_steps
        except AttributeError:
            model_name = type(self.estimator).__name__
        else:
            pipeline_steps_names = list(pipeline_steps.keys())
            last_estimator_name = pipeline_steps_names[-1]
            last_estimator = self.estimator[last_estimator_name]
            model_name = type(last_estimator).__name__

        return model_name

    @property
    def _relevant_columns(self):
        """Chooses only relevant columns.

        In this case relevant columns in the DataFrame obtained from
        `display_tuning_process()` function involve varied hyperparameters,
        test/train metric values obtained during grid search. Function
        behaves as a class property.
        """
        # Creating the DataFrame from `cv_results_` attribute
        df = pd.DataFrame(self.cv_results_)

        # Choosing specific columns
        columns_to_display = df.columns.str.startswith(("param_", "mean_t"))
        relevant_columns = df.columns[columns_to_display]

        return relevant_columns

    def _fix_col_names(self, df):
        """Fixes columns names in the DataFrame.

        Adjusts column names in the DataFrame returned
        by `display_tuning_process()` function.
        """
        try:
            pipeline_steps = self.estimator.named_steps
        except AttributeError:
            df.columns = df.columns.str.replace("param_", "")
        else:
            pipeline_steps_names = list(pipeline_steps.keys())
            last_estimator_name = pipeline_steps_names[-1]
            col_name = "param_" + last_estimator_name + "__"
            df.columns = df.columns.str.replace(col_name, "")
        finally:
            df = df.round(4)

        return df

    def _verify_est_fitted(self):
        """Checks if the estimator has been fit."""
        if not hasattr(self, "cv_results_"):
            not_fitted_error_msg = (
                f"The {type(self).__name__} instance has not been fitted yet. "
                "Call 'train_crossvalidate' before using this method."
            )
            raise NotFittedError(not_fitted_error_msg)

    @exec_timer
    def train_crossvalidate(self) -> Any:
        """Launches grid search algorithm.

        Using a custom `exec_timer` decorator,
        displaying total grid search time (in
        seconds).
        """

        self.fit(self.train_features, self.train_target)

        model_name = self._model_name

        print(f"Grid search for {model_name} completed.")

    def display_tuning_process(self) -> pd.DataFrame:
        """
        Displaying the results of grid search. Additionally, displays the
        best combination of hyperparameters and scoring metric values.
        """
        # Verifying if the grid search has been launched
        self._verify_est_fitted()

        # Creating the initial DataFrame from `cv_results_` attribute
        cv_results_df = pd.DataFrame(self.cv_results_)

        # Selecting the relevant columns
        relevant_columns = self._relevant_columns
        cv_results_df = cv_results_df[relevant_columns]

        # Fixing column names in the DataFrame
        cv_results_df = self._fix_col_names(cv_results_df)

        # Adjusting the sign in case of regression task
        cv_results_df["mean_test_score"] = -cv_results_df["mean_test_score"]

        if self.return_train_score:
            cv_results_df["mean_train_score"] = -cv_results_df[
                "mean_train_score"
            ]

        return cv_results_df

    @property
    def best_iter(self):
        """Displays the best iteration.

        Based on the DataFrame from `display_tuning_process()`,
        retrieves the best iteration (DataFrame row) and prints
        it out. Behaves as a property.
        """
        # Computing the table with results
        df = self.display_tuning_process()

        # Selecting the best index
        best_iteration = self.best_index_

        # Choosing the best row
        cv_results_best_iter = df.iloc[best_iteration]

        # Retrieving model name (last estimator name)
        model_name = self._model_name

        # Renaming the pd.Series
        cv_results_best_iter = cv_results_best_iter.rename(model_name)

        return cv_results_best_iter

    def compute_regression_stats(
        self, metric: Literal["rmse", "mse", "mae"]
    ) -> float:
        """Computes key regression metrics."""
        # Verifying if the grid search has been launched
        self._verify_est_fitted()
        # Proceed if exception has not been raised
        self.eval_predictions = self.predict(self.eval_features)
        if metric == "rmse":
            rmse = mean_squared_error(
                self.eval_target, self.eval_predictions, squared=False
            )
            return rmse
        elif metric == "mse":
            mse = mean_squared_error(
                self.eval_target, self.eval_predictions, squared=True
            )
            return mse
        elif metric == "mae":
            mae = mean_absolute_error(self.eval_target, self.eval_predictions)
            return mae
        else:
            raise ValueError("Incorrect metric name specified")


class RandomizedHyperoptClassification(RandomizedSearchCV):
    """
    Wrapper for RandomizedSearchCV with custom defaults
    and additional functionality for solving classification
    problems.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Union[Sequence[Mapping], Mapping],
        train_dataset: Tuple[Any, Any],
        eval_dataset: Tuple[Any, Any],
        cv: Union[Iterable, int] = 2,
        random_state: Union[RandomStateInstance, int] = 12345,
        scoring: str = "f1",
        n_iter: int = 5,
        n_jobs: Optional[int] = None,
        return_train_score: bool = True,
    ) -> None:
        """Initializes an instance.

        Args:
            estimator (BaseEstimator): Sklearn model, Gradient Boosting model or
                Pipeline object.
            param_distributions (Union[Sequence[Mapping], Mapping]): Grid of
                hyperparameter names and their values to be varied.
            train_dataset (Tuple[Any, Any]): Tuple of training features and
                training target-vector.
            eval_dataset (Tuple[Any, Any]): Tuple of testing features and
                testing target-vector.
            cv (Union[Iterable, int], optional): Crossvalidator represented
                as an iterator or an integer. Defaults to 2.
            random_state (Union[RandomStateInstance, int], optional): Random seed
                represented as either RandomState instance or an integer.
                Defaults to 12345.
            scoring (str, optional): Scoring metric.
                Defaults to "f1".
            n_iter (int, optional): Number of hyperparameter combinations
                to consider. Defaults to 5.
            n_jobs (Optional[int], optional): Parallelization of computations.
                Defaults to None.
            return_train_score (bool, optional): Boolean indicating returning
                metric values scores computed on training set. Defaults to True.
        """
        super().__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            cv=cv,
            random_state=random_state,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_features, self.train_target = self.train_dataset
        self.eval_features, self.eval_target = self.eval_dataset

    @property
    def _model_name(self):
        """Retrieves last estimator name.

        The function which ends up behaving as a class property
        returns the name of the last estimator in the pipeline or
        the plain model name used.
        """
        try:
            pipeline_steps = self.estimator.named_steps
        except AttributeError:
            model_name = type(self.estimator).__name__
        else:
            pipeline_steps_names = list(pipeline_steps.keys())
            last_estimator_name = pipeline_steps_names[-1]
            last_estimator = self.estimator[last_estimator_name]
            model_name = type(last_estimator).__name__

        return model_name

    @property
    def _relevant_columns(self):
        """Chooses only relevant columns.

        In this case relevant columns in the DataFrame obtained from
        `display_tuning_process()` function involve varied hyperparameters,
        test/train metric values obtained during grid search. Function
        behaves as a class property.
        """
        df = pd.DataFrame(self.cv_results_)

        # Choosing specific columns
        columns_to_display = df.columns.str.startswith(("param_", "mean_t"))
        relevant_columns = df.columns[columns_to_display]

        return relevant_columns

    def _fix_col_names(self, df):
        """Fixes columns names in the DataFrame.

        Adjusts column names in the DataFrame returned
        by `display_tuning_process()` function.
        """
        try:
            pipeline_steps = self.estimator.named_steps
        except AttributeError:
            df.columns = df.columns.str.replace("param_", "")
        else:
            pipeline_steps_names = list(pipeline_steps.keys())
            last_estimator_name = pipeline_steps_names[-1]
            col_name = "param_" + last_estimator_name + "__"
            df.columns = df.columns.str.replace(col_name, "")
        finally:
            df = df.round(4)

        return df

    def _verify_est_fitted(self):
        """Checks if the estimator has been fit."""
        if not hasattr(self, "cv_results_"):
            not_fitted_error_msg = (
                f"The {type(self).__name__} instance has not been fitted yet. "
                "Call 'train_crossvalidate' before using this method."
            )
            raise NotFittedError(not_fitted_error_msg)

    @exec_timer
    def train_crossvalidate(self) -> Any:
        """Launches grid search algorithm.

        Using a custom `exec_timer` decorator,
        displaying total grid search time (in
        seconds).
        """

        self.fit(self.train_features, self.train_target)

        model_name = self._model_name

        print(f"Grid search for {model_name} completed.")

    def display_tuning_process(self) -> pd.DataFrame:
        """
        Displaying the results of grid search. Additionally, displays the
        best combination of hyperparameters and scoring metric values.
        """
        # Verifying if the grid search has been launched
        self._verify_est_fitted()

        # Creating the initial DataFrame from `cv_results_` attribute
        cv_results_df = pd.DataFrame(self.cv_results_)

        # Selecting the relevant columns
        relevant_columns = self._relevant_columns
        cv_results_df = cv_results_df[relevant_columns]

        # Fixing column names in the DataFrame
        cv_results_df = self._fix_col_names(cv_results_df)

        return cv_results_df

    @property
    def best_iter(self):
        """Displays the best iteration.

        Based on the DataFrame from `display_tuning_process()`,
        retrieves the best iteration (DataFrame row) and prints
        it out. Behaves as a property.
        """
        # Computing the table with results
        df = self.display_tuning_process()

        # Selecting the best index
        best_iteration = self.best_index_

        # Choosing the best row
        cv_results_best_iter = df.iloc[best_iteration]

        # Retrieving model name (last estimator name)
        model_name = self._model_name

        # Renaming the pd.Series
        cv_results_best_iter = cv_results_best_iter.rename(model_name)

        return cv_results_best_iter

    def compute_classification_stats(
        self,
        target_names: Optional[Tuple[str]] = None,
        metric: Optional[str] = None,
    ) -> Optional[float]:
        """Performs computation of classification metrics.

        Args:
            target_names (Tuple[str], optional): Names of classes.
                Defaults to None.
            metric (Optional[str], optional): Name of metric.
                Defaults to None.

        Returns:
            Optional[float]: Floating point number, value of a
            specific classification metric or a classification
            metrics table.
        """
        # Verifying if the grid search has been launched
        self._verify_est_fitted()

        # Computing predictions on the test set
        self.eval_predictions = self.predict(self.eval_features)

        # Return the value of a particular metric
        NoneType = type(None)
        if not isinstance(metric, NoneType):
            # Adding ROC-AUC to metric values to be returned
            if metric == "roc-auc":
                roc_auc_value = roc_auc_score(
                    self.eval_target,
                    self.predict_proba(self.eval_features)[:, 1],
                )

                return roc_auc_value
            # Return some other value of metrics contained in classification_report
            else:
                classification_stats = classification_report(
                    self.eval_target,
                    self.eval_predictions,
                    target_names=target_names,
                    output_dict=True,
                )

                classes_list = list(classification_stats.keys())
                last_class = classes_list[1]
                metric_score = classification_stats[last_class][metric]

                return metric_score
        # Print out a summary of all classification metrics if metric=None
        classification_stats = classification_report(
            self.eval_target, self.eval_predictions, target_names=target_names
        )
        print(classification_stats)

    def plot_confusion_matrix(
        self, label_names: Optional[Tuple[str]] = None
    ) -> Any:
        """Plots a confusion matrix."""
        # Verifying if the grid search has been launched
        self._verify_est_fitted()

        # Computing predictions on the test set
        self.eval_predictions = self.predict(self.eval_features)

        # Plotting confusion matrix
        sns.reset_defaults()
        cm = confusion_matrix(self.eval_target, self.eval_predictions)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=label_names
        )
        disp.plot(cmap=plt.cm.Blues)

        # Selecting last estimator name
        model_name = self._model_name

        plt.title(f"Confusion matrix ({model_name})")
        plt.tight_layout()
        plt.show()
